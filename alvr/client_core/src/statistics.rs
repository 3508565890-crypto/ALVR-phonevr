use alvr_common::SlidingWindowAverage;
use alvr_packets::ClientStatistics;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

struct HistoryFrame {
    input_acquired: Instant,
    video_packet_received: Instant,
    client_stats: ClientStatistics,
    // Diagnostic-only marker. True when the frame was back-filled by ensure_frame (keyed by the
    // video frame timestamp) instead of created by report_input_acquired (keyed by the pose
    // prediction timestamp). Synthetic frames must never feed the latency averages used for head/
    // tracker prediction.
    synthetic: bool,
}

pub struct StatisticsManager {
    history_buffer: VecDeque<HistoryFrame>,
    max_history_size: usize,
    prev_vsync: Instant,
    total_pipeline_latency_average: SlidingWindowAverage<Duration>,
    steamvr_pipeline_latency: Duration,
}

impl StatisticsManager {
    pub fn new(
        max_history_size: usize,
        nominal_server_frame_interval: Duration,
        steamvr_pipeline_frames: f32,
    ) -> Self {
        Self {
            max_history_size,
            history_buffer: VecDeque::new(),
            prev_vsync: Instant::now(),
            total_pipeline_latency_average: SlidingWindowAverage::new(
                Duration::ZERO,
                max_history_size,
            ),
            steamvr_pipeline_latency: Duration::from_secs_f32(
                steamvr_pipeline_frames * nominal_server_frame_interval.as_secs_f32(),
            ),
        }
    }

    pub fn report_input_acquired(&mut self, target_timestamp: Duration) {
        if !self
            .history_buffer
            .iter()
            .any(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            self.history_buffer.push_front(HistoryFrame {
                input_acquired: Instant::now(),
                // this is just a placeholder because Instant does not have a default value
                video_packet_received: Instant::now(),
                client_stats: ClientStatistics {
                    target_timestamp,
                    ..Default::default()
                },
                synthetic: false,
            });
        }

        if self.history_buffer.len() > self.max_history_size {
            self.history_buffer.pop_back();
        }
    }

    // Diagnostic-only: ensure a history frame exists for the given video frame timestamp.
    // On PhoneVR/Cardboard the statistics history is keyed by the pose prediction timestamp
    // (GetBootTimeNano() + offset), which never equals the video frame timestamp passed to
    // alvr_report_submit. Without this, report_video_packet_received / report_compositor_start /
    // report_submit never find the frame and frame_interval stays 0 (server shows client_fps = 0).
    //
    // NOTE: this does NOT change decode, display, or frame submission timing. Because the frame is
    // created here (at get_frame time) after the video packet was already received, the earlier
    // report_video_packet_received and report_frame_decoded calls for this timestamp are NOT
    // back-filled, so their decode-related fields may remain 0 and must NOT be treated as real
    // decode data. The back-filled frame is marked synthetic and is excluded from the latency
    // averages used for head/tracker prediction.
    pub fn ensure_frame(&mut self, target_timestamp: Duration) {
        if !self
            .history_buffer
            .iter()
            .any(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            self.history_buffer.push_front(HistoryFrame {
                input_acquired: Instant::now(),
                video_packet_received: Instant::now(),
                client_stats: ClientStatistics {
                    target_timestamp,
                    ..Default::default()
                },
                synthetic: true,
            });
        }

        if self.history_buffer.len() > self.max_history_size {
            self.history_buffer.pop_back();
        }
    }

    pub fn report_video_packet_received(&mut self, target_timestamp: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            frame.video_packet_received = Instant::now();
        }
    }

    pub fn report_frame_decoded(&mut self, target_timestamp: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            frame.client_stats.video_decode =
                Instant::now().saturating_duration_since(frame.video_packet_received);
        }
    }

    pub fn report_compositor_start(&mut self, target_timestamp: Duration) {
        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            frame.client_stats.video_decoder_queue = Instant::now().saturating_duration_since(
                frame.video_packet_received + frame.client_stats.video_decode,
            );
        }
    }

    // vsync_queue is the latency between this call and the vsync. it cannot be measured by ALVR and
    // should be reported by the VR runtime
    pub fn report_submit(&mut self, target_timestamp: Duration, vsync_queue: Duration) {
        let now = Instant::now();

        if let Some(frame) = self
            .history_buffer
            .iter_mut()
            .find(|frame| frame.client_stats.target_timestamp == target_timestamp)
        {
            frame.client_stats.rendering = now.saturating_duration_since(
                frame.video_packet_received
                    + frame.client_stats.video_decode
                    + frame.client_stats.video_decoder_queue,
            );
            frame.client_stats.vsync_queue = vsync_queue;
            frame.client_stats.total_pipeline_latency =
                now.saturating_duration_since(frame.input_acquired) + vsync_queue;
            // Synthetic frames (back-filled by ensure_frame) must not affect head/tracker
            // prediction, so they are excluded from the latency average.
            if !frame.synthetic {
                self.total_pipeline_latency_average
                    .submit_sample(frame.client_stats.total_pipeline_latency);
            }

            let vsync = now + vsync_queue;
            frame.client_stats.frame_interval = vsync.saturating_duration_since(self.prev_vsync);
            self.prev_vsync = vsync;
        }
    }

    pub fn summary(&self, target_timestamp: Duration) -> Option<ClientStatistics> {
        self.history_buffer
            .iter()
            .find(|frame| frame.client_stats.target_timestamp == target_timestamp)
            .map(|frame| frame.client_stats.clone())
    }

    // latency used for head prediction
    pub fn average_total_pipeline_latency(&self) -> Duration {
        self.total_pipeline_latency_average.get_average()
    }

    // latency used for controllers/trackers prediction
    pub fn tracker_prediction_offset(&self) -> Duration {
        self.total_pipeline_latency_average
            .get_average()
            .saturating_sub(self.steamvr_pipeline_latency)
    }
}
