#------------------------------------------------------------------------------
#'RadioFunctions.py'                                Hearn WSU-ECE
#                                                   17apr23
# Open-Source Antenna Pattern Measurement System
#
# RadioFunctions-contains the helper functions for the radio system that are
# not part of any other class
#  
# Performs the following project-specific functions:
#   LoadParams = imports 'json' file with inputs
#   InitMotor   
#   OpenDatafile
#   rms
#   do_single
#   do_AMscan
#   do_AMmeas
#   do_NSmeas
#   get_plot_data
#   PlotFile()
#   PlotFiles()
#
#   RxRadio
#   Tx Radio
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# WSU-ECE legal statement here
#------------------------------------------------------------------------------
import numpy as np
import math
from PlotGraph import PlotGraph
import json
import RxRadio
import TxRadio
import TimeGating
from MotorController import MotorController
from PolarPlot import plot_polar_patterns, plot_patterns
import matplotlib.pyplot as plt
import time

#------------------------------------------------------------------------------
def LoadParams(filename=None):
    """ Load parameters file
        parameters are a dictionary saved in a json file
        if filename is not given then a default file will be used
        any parameters not given in the file will be used from the default file
        if the file cannot be found it will raise an exception
    """
    try:
        defaults=json.load(open("params_default.json"))
    except Exception as e:
        print("params_default.json file is missing")
        raise e 
    if filename==None:
        return defaults
    try:
        params=json.load(open(filename))
    except Exception as e:
        print("Failed to load parameter file {:s}".format(filename))
        raise e
    #--------------------------------------------------------------------------
    # print(params)
    # go through all parameters given in the params file and
    # overwrite the defaults with any that are given
    #--------------------------------------------------------------------------
    for p in defaults:
        if p in params:
            defaults[p]=params[p]
        else:
            print("Parameter {:s} not specified in {:s} using default of ".format(p,filename),defaults[p])
    #--------------------------------------------------------------------------        
    # make sure freqency is within hackrf range
    #--------------------------------------------------------------------------
    if defaults["frequency"] < 30e6 or defaults["frequency"] > 6e9:
        #raise Excpetion("Frequency {:e} out of range".format(defaults["frequency"]))
        raise Exception("Frequency {:e} out of range".format(defaults["frequency"]))
    return defaults
#------------------------------------------------------------------------------
def InitMotor(params):
    motor_controller = MotorController(
    params["usb_port"],
    params["baudrate"])
    try:
        motor_controller.connect()
        print("Success: Motor controller fully connected.")
    except Exception as e:
        print("Error: Motor controller not responding, verify connections.")
        raise e
    motor_controller.reset_orientation()
    return motor_controller
#------------------------------------------------------------------------------
def OpenDatafile(params):
    filename = time.strftime("%d-%b-%Y_%H-%M-%S")+params["filename"]
    datafile_fp = open(filename, 'w')
    datafile_fp.write(params["notes"]+"\n")
    datafile_fp.write("% Mast Angle, Arm Angle, Background RSSI, Real, Imag\n")
    return datafile_fp
#------------------------------------------------------------------------------
def rms(data):
    """ return the rms of a data vector """
    return np.sqrt(np.square(data).mean())

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def do_single(Tx=True):
    params=LoadParams()
    if Tx:
        radio_tx_graph = TxRadio.RadioFlowGraph(
            params["tx_radio_id"], 
            params["frequency"], 
            params["tx_freq_offset"])
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], 
        params["frequency"], 
        params["rx_freq_offset"],
        numSamples=10000)
    if Tx:
        radio_tx_graph.start()
    radio_rx_graph.start()
    radio_rx_graph.wait()
    if Tx:
        radio_tx_graph.stop()
    rxd=radio_rx_graph.vector_sink_0.data()
    rxd2 =radio_rx_graph.vector_sink_1.data()
    plt.plot(rxd)
    plt.show()
    return rms(rxd)
    
#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def do_singleTG(params):
    """
    Simple bench test: sweep frequencies at a fixed pointing, time-gate the sweep,
    and plot the resulting 'pattern' (single angle) just to sanity check the TG.
    """
    # build freq list (rounded + dedupe)
    freq_lin = np.linspace(params["lower_frequency"], params["upper_frequency"], params["freq_steps"])
    freq_list = np.unique(np.array([round_sig(v, 3) for v in freq_lin], dtype=float))
    Nf = int(freq_list.size)
    if Nf < 2:
        print("Increase freq_steps/span: need ≥2 frequency points for time gating.")
        return None

    # collect complex mean per frequency (static antenna, no motor)
    resp = np.zeros((1, Nf), dtype=np.complex128)
    for i, f in enumerate(freq_list):
        rx = RxRadio.RadioFlowGraph(params["rx_radio_id"], f, params["rx_freq_offset"])
        tx = TxRadio.RadioFlowGraph(params["tx_radio_id"], f, params["tx_freq_offset"])
        tx.start(); time.sleep(1.5)
        rx.start(); rx.wait()
        I = np.array(rx.vector_sink_0.data(), dtype=float)
        Q = np.array(rx.vector_sink_1.data(), dtype=float)
        L = min(I.size, Q.size)
        resp[0, i] = (I[:L] + 1j*Q[:L]).mean() if L else 0.0+0.0j
        try: rx.stop(); rx.wait()
        except: pass
        try: tx.stop(); tx.wait()
        except: pass

    # time gate the 1×Nf sweep (fs inferred from df)
    gate_width = float(params.get("tg_duration_s", 25e-9))
    tg_db = TimeGating.apply_time_gating_matrix(resp, freq_list, gate_width_s=gate_width)

    print("Time-gated (dB, peak=0):", tg_db.tolist())
    return tg_db

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def do_AMscan(params):
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params) 
    time_gating_enabled = params["time_gate"]
    print("Time gating enabled: " + str(time_gating_enabled))
    radio_tx_graph = TxRadio.RadioFlowGraph(
        params["tx_radio_id"], 
        params["frequency"], 
        params["tx_freq_offset"]) 
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], 
        params["frequency"], 
        params["rx_freq_offset"])
    AMantenna_data   = []
    radio_tx_graph.start()
    time.sleep(3)                                             # Tx latency
    print("Moving to start angle")
    motor_controller.rotate_mast(params["mast_start_angle"]);
    print("Collecting data while moving to end angle")
    radio_rx_graph.start()
    motor_controller.rotate_mast(params["mast_end_angle"]);
    radio_rx_graph.stop()
    radio_tx_graph.stop();                                    # stop Tx
    print("Finished collection, return to 0")                 #
    motor_controller.rotate_mast(0);                          # Reset AUT
    antenna_data=radio_rx_graph.vector_sink_0.data()
    if time_gating_enabled:
        antenna_data = TimeGating.print_and_return_data(antenna_data)
    n=len(antenna_data)
    print("read {:d} data_points".format(n))
    antenna_pow = np.square(antenna_data)
    numangles = params["mast_end_angle"]-params["mast_start_angle"] 
    binsize=int(n/numangles)
    print("binsize= {:d}".format(binsize))
    avg=np.zeros(numangles)
    for i in range(numangles):
        avg[i]=np.sqrt(np.square(
            antenna_data[i*binsize:(i+1)*binsize]).sum()/binsize)
    angles = range(int(params["mast_start_angle"]), int(params["mast_end_angle"]),1)
    arm_angle = np.zeros(len(avg));
    background_rssi = np.zeros(len(avg));

    plt.plot(antenna_pow)
    plt.show()
    plt.plot(avg)
    plt.show()
    print("avg {:d}".format(len(avg)),binsize)
    for i in range(len(avg)):
        datafile.write(
                str(angles[i]) + ',' + 
                str(arm_angle[i]) + ',' + 
                str(background_rssi[i]) + ',' + 
                str(avg[i]) + ','+
                str(arm_angle[i]) + '\n' #place holder
                )
        AMantenna_data.append((angles[i], arm_angle[i], 
            background_rssi[i], avg[i], arm_angle[i]))

    datafile.close();
    print("datafile closed")

    return AMantenna_data
#-----------------------------------------------------------------------------
#def do_AMscan_slow(params):
#------------------------------------------------------------------------------
def do_AMmeas(params):
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params) 
    radio_tx_graph = TxRadio.RadioFlowGraph(
        params["tx_radio_id"], 
        params["frequency"], 
        params["tx_freq_offset"]) 
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], 
        params["frequency"],
        params["rx_freq_offset"], 
        numSamples=params["rx_samples"])
    antenna_data = []

    mast_angles = np.linspace(
        params["mast_start_angle"], 
        params["mast_end_angle"], 
        params["mast_steps"])
    arm_angles = np.linspace(params["arm_start_angle"], 
        params["arm_end_angle"], 
        params["arm_steps"])
    radio_tx_graph.start()
    time.sleep(3)                         # Tx latency 
    for mast_angle in mast_angles:        # azimuth control
        for arm_angle in arm_angles:      # elevation control (under constr)
            background_rssi = 0.0
            transmission_rssi = 0.0
            #
            print("Target Mast Angle: "+str(mast_angle))
            print("Target Arm Angle: "+str(arm_angle))
            print("Moving antenna...")
            motor_controller.rotate_mast(mast_angle)
            motor_controller.rotate_arm(arm_angle)
            print("Movement complete")
            #------------------------------------------------------------------
            # transmission rssi reading
            #------------------------------------------------------------------
            print("Taking transmitted signal sample...")
            radio_rx_graph.start()
            radio_rx_graph.wait()
            #radio_rx_graph.stop()
            # get data from the receiver and reset its output vector
            data=radio_rx_graph.vector_sink_0.data()
            radio_rx_graph.vector_sink_0.reset()
            radio_rx_graph.blocks_head_0.reset()
            #------------------------------------------------------------------
            #originally trimmed like this NW
            #data_points = delete(data_points, range(399000));
            #------------------------------------------------------------------
            print("read {:d} data_points".format(len(data)))
            transmission_rssi=np.sqrt(np.square(data).mean())
            print("Transmission RSSI: {:.3e}".format(transmission_rssi))
            print("Saving samples")
            datafile.write(
                str(mast_angle) + ',' + 
                str(arm_angle) + ',' + 
                str(background_rssi) + ',' + 
                str(transmission_rssi) + '\n'
                )
            antenna_data.append((mast_angle, arm_angle, 
                background_rssi, transmission_rssi))
    print("Returning mast and arm to home position...")
    motor_controller.rotate_mast(0)
    motor_controller.rotate_arm(0)
    print("Mast and arm should now be in home position")
    datafile.close();
    print("datafile closed")
    print("Scan completed")
    radio_tx_graph.stop()
    radio_tx_graph.wait()
    #
    return antenna_data
    
#==============================================================================
def do_AMTGmeas(params):
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)
    antenna_data = []

    mast_angles = np.linspace(params["mast_start_angle"],
                              params["mast_end_angle"],
                              params["mast_steps"], dtype = float)
    arm_angles = np.linspace(params["arm_start_angle"],
                             params["arm_end_angle"],
                             params["arm_steps"], dtype = float)
    
    freq_lin = np.linspace(params["lower_frequency"],
                           params["upper_frequency"],
                           params["freq_steps"], dtype = float)
    
    freq_rounded = np.array([round_sig(v,3) for v in freq_lin], dtype = float)
    freq_list = np.unique(freq_rounded)
    
    for freq in freq_list: 
        radio_rx_graph = RxRadio.RadioFlowGraph(params["rx_radio_id"], freq,
                                                params["rx_freq_offset"],
                                                numSamples = params["rx_samples"])
        radio_tx_graph = TxRadio.RadioFlowGraph(params["tx_radio_id"], freq,
                                                params["tx_freq_offset"])
        
        radio_tx_graph.start()
        time.sleep(3)
        
        for mast_angle in mast_angles:
            for arm_angle in arm_angles:
                background_rssi = 0.0

                print(f"Target Mast Angle:{mast_angle}")
                print(f"Target Arm Angle: {arm_angle}")
                motor_controller.rotate_mast(mast_angle)
                motor_controller.rotate_arm(arm_angle)
                print("Movement Complete")

                print("Capturing signal")
                radio_rx_graph.start()
                radio_rx_graph.wait()

                I = np.array(radio_rx_graph.vector_sink_0.data(), dtype = float)
                Q = np.array(radio_rx_graph.vector_sink_1.data(), dtype = float)
                L = min(I.size, Q.size)
                print(f"read {L:d} samples")

                if L == 0:
                    transmission_rssi = 0.0
                    rssi_complex = complex(0.0,0.0)
                else:
                    v_c = I[:L] + 1j*Q[:L]

                    transmission_rssi = float(np.sqrt(np.mean(np.abs(v_c)**2)))

                    rssi_complex = complex(np.mean(v_c))

                radio_rx_graph.vector_sink_0.reset(); radio_rx_graph.blocks_head_0.reset()
                radio_rx_graph.vector_sink_1.reset(); radio_rx_graph.blocks_head_1.reset()

                print(f"Transmission RSSI: {transmission_rssi:.3e}")
                print("Saving Samples")

                datafile.write(f"{mast_angle},{arm_angle},{background_rssi},"
                            f"{rssi_complex.real},{rssi_complex.imag}\n")
                antenna_data.append((mast_angle, arm_angle,
                                     background_rssi, transmission_rssi,rssi_complex))
            
            print("Returning mast to home")
            motor_controller.rotate_mast(0)
            motor_controller.rotate_arm(0)

            radio_tx_graph.stop()
            radio_tx_graph.wait()
        datafile.close()
        print("scan complete")
        return antenna_data

#===============================================================================
def do_AMTGscan(params):
    """
    Angle scan across a *frequency sweep*, then time-gate to remove reflections.
    Produces a polar plot and writes a *_TG.csv file with the gated linear magnitudes.
    """
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)  # raw capture file (complex per angle written below)

    # --- build freq list (rounded & deduped) ---
    freq_lin = np.linspace(float(params["lower_frequency"]),
                           float(params["upper_frequency"]),
                           int(params["freq_steps"]), dtype=float)
    freq_list = np.unique(np.array([round_sig(v, 3) for v in freq_lin], dtype=float))
    Nf = int(freq_list.size)
    if Nf < 2:
        print("WARNING: time gating needs multiple frequencies. Increase freq_steps or span.")
    print(f"Using {Nf} frequency points.")

    # --- angle grid ---
    mast_start = float(params["mast_start_angle"])
    mast_end   = float(params["mast_end_angle"])
    mast_steps = int(params["mast_steps"])
    mast_angles = np.linspace(mast_start, mast_end, mast_steps, endpoint=False, dtype=float)

    # storage: complex response per angle per freq
    responses = np.zeros((mast_steps, Nf), dtype=np.complex128)

    # --- sweep ---
    for fi, freq in enumerate(freq_list):
        rx = RxRadio.RadioFlowGraph(params["rx_radio_id"], freq, params["rx_freq_offset"])
        tx = TxRadio.RadioFlowGraph(params["tx_radio_id"], freq, params["tx_freq_offset"])

        tx.start()
        time.sleep(3)  # TX latency

        print(f"[{fi+1}/{Nf}] freq={freq/1e6:.3f} MHz: rotating & collecting …")
        per_angle = _collect_complex_per_angle(rx, mast_start, mast_end, mast_steps, motor_controller)
        responses[:, fi] = per_angle

        tx.stop(); tx.wait()
        motor_controller.rotate_mast(0)

        # also log *complex* mean per angle (raw) to the open file
        for ang, cval in zip(mast_angles, per_angle):
            datafile.write(f"{ang:.1f},0.0,0.0,{cval.real:.8e},{cval.imag:.8e}\n")

    datafile.close()
    print("raw datafile closed")

    # --- time gating ---
    gate_width = float(params.get("tg_duration_s", 25e-9))
    print(f"Applying time gating with width {gate_width*1e9:.1f} ns …")
    gated_db = TimeGating.apply_time_gating_matrix(
    responses, freq_list, gate_width_s=gate_width,
    denoise_wavelet=True
)


    # --- plot polar + cartesian (optional) ---
    deg = mast_angles
    plot_polar_patterns(
        deg,
        traces=[("Time-Gated", gated_db)],
        rmin=-60.0, rmax=0.0, rticks=(-60, -40, -20, 0),
        title="Radiation Pattern (Time-Gated, Polar)"
    )
    # cartesian overlay (nice for sanity)
    try:
        plot_patterns(deg, traces=[("Time-Gated", gated_db)], title="Radiation Pattern (Time-Gated)")
    except Exception:
        pass

    # --- write gated results (linear magnitude, normalized) ---
    y_lin = 10.0 ** (gated_db / 20.0)
    outname = time.strftime("%d-%b-%Y_%H-%M-%S") + "_TG.csv"
    with open(outname, "w") as fp:
        fp.write("Angle,Arm,Background,GatedMag\n")
        for a, v in zip(deg, y_lin):
            fp.write(f"{a:.1f},0.0,0.0,{v:.8e}\n")
    print(f"wrote gated results → {outname}")

    return list(zip(deg.tolist(), [0.0]*len(deg), [0.0]*len(deg), y_lin.tolist()))

def do_AMTGscan_single_freq(params, freq_hz: float = 2.5e9, *, show_plots: bool = True):
    import TimeGating
    from PolarPlot import plot_polar_patterns
    import numpy as np, time

    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)

    # one frequency (rounded to 3 sig figs for the SDR)
    freq = float(round_sig(freq_hz, 3))

    mast_start = float(params["mast_start_angle"])
    mast_end   = float(params["mast_end_angle"])
    mast_steps = int(params["mast_steps"])
    mast_angles = np.linspace(mast_start, mast_end, mast_steps, endpoint=False, dtype=float)

    rx = tx = None
    per_angle_complex = None
    try:
        rx = RxRadio.RadioFlowGraph(params["rx_radio_id"], freq, params["rx_freq_offset"])
        tx = TxRadio.RadioFlowGraph(params["tx_radio_id"], freq, params["tx_freq_offset"])

        tx.start()
        time.sleep(3)

        print(f"Single-frequency scan @ {freq/1e9:.3f} GHz: rotating & collecting …")
        per_angle_complex = _collect_complex_per_angle(rx, mast_start, mast_end, mast_steps, motor_controller)

    finally:
        # cleanup
        try:
            if rx: rx.stop(); rx.wait()
        except Exception: pass
        try:
            if tx: tx.stop(); tx.wait()
        except Exception: pass
        try:
            motor_controller.rotate_mast(0)
        except Exception: pass

    # write raw complex per-angle data
    for ang, cval in zip(mast_angles, per_angle_complex):
        datafile.write(f"{ang:.1f},0.0,0.0,{cval.real:.8e},{cval.imag:.8e}\n")
    datafile.close()
    print("raw datafile closed")

    # --- Build patterns ---
    # noisy_db = 20*np.log10(np.abs(per_angle_complex)/np.max(np.abs(per_angle_complex)))
    noisy_mag = np.abs(per_angle_complex)
    noisy_mag /= (np.max(noisy_mag) if np.max(noisy_mag) > 0 else 1.0)
    noisy_db = 20*np.log10(np.clip(noisy_mag, 1e-12, None))


    # shape it like [angles, freqs] with 1 frequency column
    freq_resp = per_angle_complex[:, np.newaxis]

    fs = float(params.get("fs", 200e6))   # sample rate
    N_fft = 2048                          # tunable

    h_t = TimeGating.impulse_response(freq_resp, N_fft)
    h_t_gated = TimeGating.apply_time_gate(h_t, fs, gate_ns=10.0, alpha=0.5)
    H_gated = TimeGating.gated_frequency_response(h_t_gated, N_fft)
    gated_pattern = TimeGating.extract_pattern(H_gated)
    gated_pattern = TimeGating.denoise_pattern(gated_pattern)
    gated_pattern -= np.max(gated_pattern)
    

    # --- Plot ---
    if show_plots:
        plot_polar_patterns(
            mast_angles,
            traces=[
                ("Raw (noisy)", noisy_db),
                ("Time-Gated", gated_pattern)
            ],
            rmin=-60, rmax=0, rticks=(-60, -40, -20, 0),
            title=f"Noisy vs Time-Gated Pattern @ {freq/1e9:.2f} GHz"
        )

    return {
        "angles_deg": mast_angles.tolist(),
        "complex_per_angle": per_angle_complex.tolist(),
        "noisy_db": noisy_db.tolist(),
        "tg_db": gated_pattern.tolist(),
    }


        
#==============================================================================
#------------------------------------------------------------------------------
# non-coherent noise-subtraction method (1st algorithm)
def do_NSmeas(params):
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params) 
    radio_tx_graph = TxRadio.RadioFlowGraph(
        params["tx_radio_id"], 
        params["frequency"], 
        params["tx_freq_offset"]) 
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], 
        params["frequency"], 
        params["rx_freq_offset"], 
        numSamples=params["rx_samples"])
    antenna_data = []
    mast_angles = np.linspace(
        params["mast_start_angle"], 
        params["mast_end_angle"], 
        params["mast_steps"])
    arm_angles  = np.linspace(
        params["arm_start_angle"], 
        params["arm_end_angle"], 
        params["arm_steps"])

    for mast_angle in mast_angles:                           # azimuth
         for arm_angle in arm_angles:                        # elevation

             background_rssi = 0.0
             transmission_rssi = 0.0

             print("Target Mast Angle: "+str(mast_angle))
             print("Target Arm Angle: "+str(arm_angle))
             print("Moving antenna...")
             motor_controller.rotate_mast(mast_angle)
             motor_controller.rotate_arm(arm_angle)
             print("Movement complete")
             print("Taking background noise sample...")       # bkgrnd rssi 
             radio_rx_graph.start()
             radio_rx_graph.wait()
             #-----------------------------------------------------------------
             # get data from the receiver and reset its output vector
             # TODO the other scans use RMS, this just does average? 
             #-----------------------------------------------------------------
             data=radio_rx_graph.vector_sink_0.data()
             print("received {:d} background samples".format(len(data)))
             radio_rx_graph.vector_sink_0.reset()
             radio_rx_graph.blocks_head_0.reset()
             background_rssi = rms(data)
             #-----------------------------------------------------------------
             # Transmission rssi reading
             #-----------------------------------------------------------------
             print("Taking transmitted signal sample...")
             radio_tx_graph.start()
             time.sleep(1.3)                                  # Tx latency
             radio_rx_graph.start()
             radio_rx_graph.wait()
             radio_tx_graph.stop()
             radio_tx_graph.wait()
             # get data from the receiver and reset its output vector
             data=radio_rx_graph.vector_sink_0.data()
             print("received {:d} transmitted samples".format(len(data)))
             radio_rx_graph.vector_sink_0.reset()
             radio_rx_graph.blocks_head_0.reset()
             #-----------------------------------------------------------------
             # TODO the other scans use RMS, this just does average? 
             #-----------------------------------------------------------------
             transmission_rssi = rms(data)
             #-----------------------------------------------------------------
             # write rssi readings to file print("Saving samples")
             #-----------------------------------------------------------------
             datafile.write(
                 str(mast_angle) + ',' + 
                 str(arm_angle) + ',' + 
                 str(background_rssi) + ',' + 
                 str(transmission_rssi) + '\n'
                 )
             print("Sample angle={:f} bkgnd={:e} received={:e}".format(
                 mast_angle, background_rssi,transmission_rssi))
             antenna_data.append((mast_angle, arm_angle, 
                background_rssi, transmission_rssi))
    print("Returning mast and arm to home position...")
    motor_controller.rotate_mast(0)
    motor_controller.rotate_arm(0)
    print("Mast and arm should now be in home position")
    datafile.close()
    return antenna_data
#------------------------------------------------------------------------------
# plot functions for menu
#------------------------------------------------------------------------------
def get_plot_data(text):
    dataPoint = 0
    fileData = []
    for dataString in text:
        dataPointString = ''
        dataTuple = []
        for tempChar in dataString:
            if tempChar == ',' or tempChar == '\n':
                dataPoint = float(dataPointString)
                dataTuple.append(dataPoint)
                dataPointString = ''
            else:
                dataPointString += tempChar
        fileData.append((dataTuple[0],dataTuple[1],dataTuple[2],dataTuple[3]))
    return fileData;
def PlotFile():
    fileName = input("Enter the name of the data to plot\n")
    fr = open(fileName)
    text = fr.readlines()
    fr.close()
    text.remove(text[0])
    text.remove(text[0])
    fileData = get_plot_data(text);
    plot_graph = PlotGraph(fileData, fileName)
    plot_graph.show()
def PlotFiles():
    fileName = input("Enter the name of first file to plot\n")
    fr = open(fileName)
    text = fr.readlines()
    fr.close()
    text.remove(text[0])
    text.remove(text[0])
    fileData = get_plot_data(text);
    plot_graph1 = PlotGraph(fileData, fileName)
    fileName = input("Enter the name of the second file to plot\n")
    fr = open(fileName)
    text = fr.readlines()
    fr.close()
    text.remove(text[0])
    text.remove(text[0])
    fileData = get_plot_data(text);
    plot_graph2 = PlotGraph(fileData, fileName)
    ax1 = plt.subplot(111, projection='polar')
    ax1.set_theta_zero_location("N")
    theta1 = [angle*(np.pi/180) for angle in plot_graph1.mast_angles]
    ax1.plot(theta1, plot_graph1.rssi, label="With Gain")
    ax2 = plt.subplot(111, projection='polar')
    ax2.set_theta_zero_location("N")
    theta2 = [angle*(np.pi/180) for angle in plot_graph2.mast_angles]
    ax2.plot(theta2, plot_graph2.rssi, label="No Gain", linewidth=1)
#    
    if plot_graph1.plot_in_db == 'y':
        ax1.set_rticks([-20,-16,-12,-8,-4,0]);
        ax2.set_rticks([-20,-16,-12,-8,-4,0]);
    plt.legend(loc="lower center", bbox_to_anchor=(1, 1))
    plt.show()



#helper functions
def _tg_data_to_dB(td: np.ndarray, pick: str = "max", idx: int | None = None) -> np.ndarray:
    mag = np.abs(td)
    if pick == "center":
        if idx is None:
            idx = mag.shape[1] // 2
        y = mag[:, idx]
    else:
        y = mag.max(axis=1)
    y = y/ (np.max(y) if np.max(y)>0 else 1.0)
    return 20.0*np.log10(np.clip(y,1e-12, None))


def round_sig(x: float, sig: int = 3) -> float:
    if not np.isfinite(x) or x == 0.0:
        return 0.0 if x == 0.0 else x
    ndigits = int(sig - 1 - np.floor(np.log10(abs(x))))
    ndigits = max(-12, min(12, ndigits))
    return round(x, ndigits)



def _collect_complex_per_angle(rx_graph,
                               mast_start: float,
                               mast_end: float,
                               mast_steps: int,
                               motor_controller) -> np.ndarray:
    """
    Start RX, rotate mast from start->end, stop RX.
    Return complex mean per angle bin: shape (mast_steps,)
    """
    motor_controller.rotate_mast(mast_start)

    rx_graph.start()
    motor_controller.rotate_mast(mast_end)
    rx_graph.stop(); rx_graph.wait()

    I = np.array(rx_graph.vector_sink_0.data(), dtype=float)
    Q = np.array(rx_graph.vector_sink_1.data(), dtype=float)
    L = min(I.size, Q.size)
    if L == 0:
        return np.zeros(mast_steps, dtype=np.complex128)

    samps = I[:L] + 1j * Q[:L]

    # bin uniformly into mast_steps chunks
    idx = np.linspace(0, L, mast_steps + 1, dtype=int)
    out = np.zeros(mast_steps, dtype=np.complex128)
    for i in range(mast_steps):
        s, e = idx[i], idx[i+1]
        seg = samps[s:e] if e > s else samps[s:s+1]
        out[i] = seg.mean()
    return out
#--------------------------------------------------------------------------EoF

