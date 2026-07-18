use alvr_common::SlidingWindowAverage;
use alvr_packets::ClientStatistics;
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicU64, Ordering},
    sync::Arc,
    sync::OnceLock,
    time::{Duration, Instant},
};
// Per-second aggregated diagnostics
#[derive(Default)]
pub struct FpsDiagnostics {
    pub received: std::sync::atomic::AtomicU64,
    pub decoded: std::sync::atomic::AtomicU64,
    pub get_frame: std::sync::atomic::AtomicU64,
    pub compositor_start: std::sync::atomic::AtomicU64,
    pub submit: std::sync::atomic::AtomicU64,
    pub repeated_frame: std::sync::atomic::AtomicU64,
    pub timestamp_duplicate: std::sync::atomic::AtomicU64,
    pub timestamp_gap_over_50ms: std::sync::atomic::AtomicU64,
    pub queue_len_max: std::sync::atomic::AtomicU64,
    pub max_gap_ms: std::sync::atomic::AtomicU64,
}

// Global diagnostics instance for cross-module access
static DIAGNOSTICS: OnceLock<std::sync::Arc<FpsDiagnostics>> = OnceLock::new();
pub fn get_diagnostics_arc() -> std::sync::Arc<FpsDiagnostics> {
    DIAGNOSTICS
        .get_or_init(|| std::sync::Arc::new(FpsDiagnostics::default()))
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
    last_log_time: Instant,
    last_display_timestamp: Option<Duration>,
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
            last_log_time: Instant::now(),
            last_display_timestamp: None,
            total_pipeline_latency_average: SlidingWindowAverage::new(
                Duration::ZERO,
                max_history_size,
            ),
            steamvr_pipeline_latency: Duration::from_secs_f32(
                steamvr_pipeline_frames * nominal_server_frame_interval.as_secs_f32(),
            ),
        }
    }

    fn maybe_log_diagnostics(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_log_time) >= Duration::from_secs(1) {
            self.log_diagnostics();
            self.last_log_time = now;
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

        get_diagnostics_arc().received.fetch_add(1, Ordering::Relaxed);
        self.maybe_log_diagnostics();
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

        get_diagnostics_arc().decoded.fetch_add(1, Ordering::Relaxed);
        self.maybe_log_diagnostics();
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

        get_diagnostics_arc().compositor_start.fetch_add(1, Ordering::Relaxed);
        self.maybe_log_diagnostics();
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
            // Track timestamp differences for diagnostics
            if let Some(last_ts) = self.last_display_timestamp {
                let gap = target_timestamp.saturating_sub(last_ts);
                let gap_ms = gap.as_millis() as u64;
                if gap_ms == 0 {
                    get_diagnostics_arc().timestamp_duplicate.fetch_add(1, Ordering::Relaxed);
                } else if gap_ms > 50 {
                    get_diagnostics_arc().timestamp_gap_over_50ms.fetch_add(1, Ordering::Relaxed);
                }
                get_diagnostics_arc().max_gap_ms.fetch_max(gap_ms, Ordering::Relaxed);
            }
            self.last_display_timestamp = Some(target_timestamp);
        }

        get_diagnostics_arc().submit.fetch_add(1, Ordering::Relaxed);
        self.maybe_log_diagnostics();
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
self.total_pipeline_latency_average
            .get_average()
            .saturating_sub(self.steamvr_pipeline_latency)
    }
}

impl StatisticsManager {
    fn log_diagnostics(&self) {
        let diagnostics = get_diagnostics_arc();
        let r = diagnostics.received.swap(0, Ordering::Relaxed);
        let d = diagnostics.decoded.swap(0, Ordering::Relaxed);
        let g = diagnostics.get_frame.swap(0, Ordering::Relaxed);
        let c = diagnostics.compositor_start.swap(0, Ordering::Relaxed);
        let s = diagnostics.submit.swap(0, Ordering::Relaxed);
        let rep = diagnostics.repeated_frame.swap(0, Ordering::Relaxed);
        let td = diagnostics.timestamp_duplicate.swap(0, Ordering::Relaxed);
        let tg = diagnostics.timestamp_gap_over_50ms.swap(0, Ordering::Relaxed);
        let ql = diagnostics.queue_len_max.swap(0, Ordering::Relaxed);
        let mg = diagnostics.max_gap_ms.swap(0, Ordering::Relaxed);
        info!(
            "[PHONEVR-FPS-DIAG] recv={} dec={} get={} comp={} sub={} rep={} dup={} gap50ms={} qmax={} gapmax_ms={}",
            r, d, g, c, s, rep, td, tg, ql, mg
        );
    }
}

impl FpsDiagnostics {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    pub fn reset(&self) {
        self.received.store(0, Ordering::Relaxed);
        self.decoded.store(0, Ordering::Relaxed);
        self.get_frame.store(0, Ordering::Relaxed);
        self.compositor_start.store(0, Ordering::Relaxed);
        self.submit.store(0, Ordering::Relaxed);
        self.repeated_frame.store(0, Ordering::Relaxed);
        self.timestamp_duplicate.store(0, Ordering::Relaxed);
        self.timestamp_gap_over_50ms.store(0, Ordering::Relaxed);
        self.queue_len_max.store(0, Ordering::Relaxed);
        self.max_gap_ms.store(0, Ordering::Relaxed);
    }
}