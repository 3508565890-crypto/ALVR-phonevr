use alvr_common::{info, parking_lot::Mutex, SlidingWindowAverage};
use alvr_packets::ClientStatistics;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

pub struct FpsDiagnostics {
    received: AtomicU64,
    decoded: AtomicU64,
    get_frame: AtomicU64,
    compositor_start: AtomicU64,
    submit: AtomicU64,
    timestamp_duplicate: AtomicU64,
    timestamp_gap_over_50ms: AtomicU64,
    queue_len_max: AtomicU64,
    max_gap_ms: AtomicU64,
    last_log_time: Mutex<Instant>,
    last_submit_timestamp: Mutex<Option<Duration>>,
}

impl FpsDiagnostics {
    fn new() -> Self {
        Self {
            received: AtomicU64::new(0),
            decoded: AtomicU64::new(0),
            get_frame: AtomicU64::new(0),
            compositor_start: AtomicU64::new(0),
            submit: AtomicU64::new(0),
            timestamp_duplicate: AtomicU64::new(0),
            timestamp_gap_over_50ms: AtomicU64::new(0),
            queue_len_max: AtomicU64::new(0),
            max_gap_ms: AtomicU64::new(0),
            last_log_time: Mutex::new(Instant::now()),
            last_submit_timestamp: Mutex::new(None),
        }
    }

    pub fn report_received(&self) {
        self.received.fetch_add(1, Ordering::Relaxed);
    }

    pub fn report_decoded(&self) {
        self.decoded.fetch_add(1, Ordering::Relaxed);
    }

    pub fn report_get_frame(&self) {
        self.get_frame.fetch_add(1, Ordering::Relaxed);
    }

    pub fn report_compositor_start(&self) {
        self.compositor_start.fetch_add(1, Ordering::Relaxed);
    }

    pub fn report_submit(&self, target_timestamp: Duration) {
        self.submit.fetch_add(1, Ordering::Relaxed);

        let mut last_submit_timestamp = self.last_submit_timestamp.lock();
        if let Some(previous_timestamp) = *last_submit_timestamp {
            if target_timestamp == previous_timestamp {
                self.timestamp_duplicate.fetch_add(1, Ordering::Relaxed);
            } else if target_timestamp > previous_timestamp {
                let gap = target_timestamp.saturating_sub(previous_timestamp);
                let gap_ms = gap.as_millis() as u64;

                self.max_gap_ms.fetch_max(gap_ms, Ordering::Relaxed);
                if gap > Duration::from_millis(50) {
                    self.timestamp_gap_over_50ms
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        *last_submit_timestamp = Some(target_timestamp);
    }

    pub fn update_queue_len_max(&self, queue_len: u64) {
        self.queue_len_max.fetch_max(queue_len, Ordering::Relaxed);
    }

    fn log_interval(&self) {
        let now = Instant::now();
        let mut last_log_time = self.last_log_time.lock();
        if now.saturating_duration_since(*last_log_time) < Duration::from_secs(1) {
            return;
        }
        *last_log_time = now;
        drop(last_log_time);

        let received = self.received.swap(0, Ordering::Relaxed);
        let decoded = self.decoded.swap(0, Ordering::Relaxed);
        let get_frame = self.get_frame.swap(0, Ordering::Relaxed);
        let compositor_start = self.compositor_start.swap(0, Ordering::Relaxed);
        let submit = self.submit.swap(0, Ordering::Relaxed);
        let timestamp_duplicate = self.timestamp_duplicate.swap(0, Ordering::Relaxed);
        let timestamp_gap_over_50ms = self
            .timestamp_gap_over_50ms
            .swap(0, Ordering::Relaxed);
        let queue_len_max = self.queue_len_max.swap(0, Ordering::Relaxed);
        let max_gap_ms = self.max_gap_ms.swap(0, Ordering::Relaxed);

        // unsupported: current client does not reuse old frames
        info!(
            "[PHONEVR-FPS-DIAG] recv={} dec={} get={} comp={} sub={} rep=unsupported dup={} gap50ms={} qmax={} gapmax_ms={}",
            received,
            decoded,
            get_frame,
            compositor_start,
            submit,
            timestamp_duplicate,
            timestamp_gap_over_50ms,
            queue_len_max,
            max_gap_ms,
        );
    }
}

static DIAGNOSTICS: OnceLock<Arc<FpsDiagnostics>> = OnceLock::new();

pub fn get_diagnostics_arc() -> Arc<FpsDiagnostics> {
    DIAGNOSTICS
        .get_or_init(|| {
            let diagnostics = Arc::new(FpsDiagnostics::new());
            let logger_diagnostics = Arc::clone(&diagnostics);

            let _ = thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(1));
                logger_diagnostics.log_interval();
            });

            diagnostics
        })
        .clone()
}

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

        get_diagnostics_arc().report_received();
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

        get_diagnostics_arc().report_decoded();
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

        get_diagnostics_arc().report_compositor_start();
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

        get_diagnostics_arc().report_submit(target_timestamp);
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
