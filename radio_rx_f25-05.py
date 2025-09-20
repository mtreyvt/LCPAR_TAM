# RadioFlowGraphIQ.py  — complex-IQ capture version
from gnuradio import analog
from gnuradio import blocks
from gnuradio import filter
from gnuradio import gr
from gnuradio.filter import firdes, window
import osmosdr

class RadioFlowGraph(gr.top_block):
    def __init__(self, radio_id, frequency, freq_offset_hz=0.0,
                 numSamples=None, simulation=False):
        """
        radio_id: HackRF serial as hex (e.g., '61555f')
        frequency: RF center frequency (Hz)
        freq_offset_hz: extra fine mixer shift after tuning (Hz). Use 0 if not needed.
        numSamples: number of complex samples to capture to vector_sink_c
        simulation: if True, uses a vector_source_c instead of osmosdr
        """
        gr.top_block.__init__(self)

        self.radio_id = radio_id
        self.frequency = float(frequency)
        self.freq_offset_hz = float(freq_offset_hz)
        self.simulation = simulation

        # --- Radio/sample-rate plumbing ---
        # Use 10 MS/s to match your current source setup
        self.radio_sample_rate = 10e6

        # Two-stage decimation to keep FIR sizes reasonable:
        # 10e6 -> 100 kS/s (decim 100), then 100 kS/s -> 2 kS/s (decim 50)
        self.decim1 = 100
        self.decim2 = 50
        self.fs1 = self.radio_sample_rate / self.decim1   # 100 kS/s
        self.final_sample_rate = self.fs1 / self.decim2   # 2 kS/s

        # ---- Source (HackRF or sim) ----
        if not self.simulation:
            src = "numchan=1 hackrf=" + self.radio_id
            self.osmosdr_source_0 = osmosdr.source(args=src)
            self.osmosdr_source_0.set_sample_rate(self.radio_sample_rate)
            self.osmosdr_source_0.set_clock_source('external', 0)
            self.osmosdr_source_0.set_time_source('internal', 0)
            self.osmosdr_source_0.set_center_freq(self.frequency, 0)
            self.osmosdr_source_0.set_freq_corr(0, 0)
            self.osmosdr_source_0.set_dc_offset_mode(0, 0)
            self.osmosdr_source_0.set_iq_balance_mode(0, 0)
            self.osmosdr_source_0.set_gain_mode(False, 0)
            self.osmosdr_source_0.set_gain(0, 0)
            self.osmosdr_source_0.set_if_gain(10, 0)
            self.osmosdr_source_0.set_bb_gain(10, 0)
            self.osmosdr_source_0.set_antenna('', 0)
            self.osmosdr_source_0.set_bandwidth(50e3, 0)  # front-end BW hint
        else:
            self.vector_source_0 = blocks.vector_source_c([])

        # ---- Optional fine mixer (complex LO) ----
        # If freq_offset_hz == 0, this still works (acts as 1.0 + 0j)
        self.analog_sig_source_lo = analog.sig_source_c(
            self.radio_sample_rate, analog.GR_COS_WAVE, -self.freq_offset_hz, 1.0, 0.0
        )
        self.mixer = blocks.multiply_vcc(1)

        # ---- Complex low-pass + decimation stages ----
        # Stage 1: decimate 100 → 100 kS/s
        # Choose a reasonably narrow passband relative to the final signal BW.
        self.lpf1 = filter.fir_filter_ccf(
            self.decim1,
            firdes.low_pass(
                gain=1.0,
                sampling_freq=self.radio_sample_rate,
                cutoff_freq=50e3,          # keep desired BW
                transition_width=10e3,
                window=window.WIN_HAMMING,
                beta=6.76,
            ),
        )

        # Stage 2: decimate 50 → 2 kS/s
        self.lpf2 = filter.fir_filter_ccf(
            self.decim2,
            firdes.low_pass(
                gain=1.0,
                sampling_freq=self.fs1,
                cutoff_freq=8e3,           # adapt as needed
                transition_width=2e3,
                window=window.WIN_HAMMING,
                beta=6.76,
            ),
        )

        # ---- Complex DC blocker ----
        self.dc_blocker = filter.dc_blocker_cc(32, True)

        # ---- Capture: complex head + sink ----
        self.numSamples = int(numSamples) if numSamples is not None else 2000
        self.head_c = blocks.head(gr.sizeof_gr_complex, self.numSamples)
        self.vector_sink_c = blocks.vector_sink_c(1, self.numSamples)

        # ================= Connections =================
        if not self.simulation:
            self.connect(self.osmosdr_source_0, self.mixer, self.lpf1, self.lpf2, self.dc_blocker, self.head_c, self.vector_sink_c)
            self.connect(self.analog_sig_source_lo, (self.mixer, 1))
        else:
            self.connect(self.vector_source_0, self.mixer, self.lpf1, self.lpf2, self.dc_blocker, self.head_c, self.vector_sink_c)
            self.connect(self.analog_sig_source_lo, (self.mixer, 1))

        # Expose convenient accessors
        self.get_iq = lambda: self.vector_sink_c.data()     # tuple of complex samples
        self.get_fs = lambda: self.final_sample_rate        # 2 kS/s here
