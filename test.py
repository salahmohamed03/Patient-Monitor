import sys
import numpy as np 
from scipy import signal
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtMultimedia import QSound
from scipy.signal import welch

# Don't import pyqtgraph or initialize any widgets at the module level
# Import wfdb in the main function

class ArrhythmiaDetector:
    """Class to detect different types of arrhythmias from ECG signal"""
    
    def __init__(self):
        self.tachycardia_threshold = 100  # BPM
        self.bradycardia_threshold = 50   # BPM
        self.afib_threshold = 0.2      # Irregularity index threshold
        
    def detect_heart_rate(self, ecg_signal, fs=250):
        """Calculate heart rate from ECG signal"""
        # Simple R-peak detection using thresholding
        filtered = signal.savgol_filter(ecg_signal, 31, 3)  # Smooth the signal
        
        # Find peaks (R waves)
        r_peaks, _ = signal.find_peaks(filtered, height=0.5*np.max(filtered), distance=fs*0.5)
        
        if len(r_peaks) < 2:
            return 0  # Not enough peaks to calculate heart rate
        
        # Calculate average RR interval and convert to BPM
        rr_intervals = np.diff(r_peaks) / fs  # in seconds
        self.mean_hr = 60 / np.mean(rr_intervals)  # Convert to BPM
        
        return self.mean_hr

    def detect_dominant_frequency(self, ecg_signal, fs):
        """Compute the dominant frequency of the ECG signal using Welch’s method"""
        freqs, power = welch(ecg_signal, fs=fs)
        peak_freq = freqs[np.argmax(power)]
        return peak_freq

    def detect_tachycardia(self, heart_rate):
        """Detect tachycardia based on heart rate"""
        return heart_rate > self.tachycardia_threshold
    
    def detect_bradycardia(self, heart_rate):
        """Detect bradycardia based on heart rate"""
        return 0 < heart_rate < self.bradycardia_threshold

    def detect_afib(self, ecg_signal, fs=250):
        """
        Detect atrial fibrillation based on RR interval irregularity
        Returns: True if AFib is detected, False otherwise
        """
        # This is a simplified method - real AFib detection is more complex
        filtered = signal.savgol_filter(ecg_signal, 31, 3)
        r_peaks, _ = signal.find_peaks(filtered, height=0.5*np.max(filtered), distance=fs*0.5)
        
        if len(r_peaks) < 10 or self.mean_hr < 50:  # Need sufficient peaks for analysis
            return False
            
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / fs  # in seconds
        
        # Calculate irregularity index (standard deviation / mean)
        irregularity = np.std(rr_intervals) / np.mean(rr_intervals)
        
        return irregularity > self.afib_threshold

        # peak_freq = self.detect_dominant_frequency(ecg_signal, fs)
        # return peak_freq > 6

class ECGMonitoringSystem(QtWidgets.QMainWindow):
    """Main application window for ECG Monitoring System"""
    
    def __init__(self, pg):
        super().__init__()
        
        # Load UI from file
        uic.loadUi('MainWindow.ui', self)
        
        # Initialize detector
        self.detector = ArrhythmiaDetector()
        self.last_valid_hr = None

        
        # Replace QGraphicsView with pyqtgraph PlotWidget
        self.ecgPlot = pg.PlotWidget()
        self.verticalLayout_2.replaceWidget(self.ecgGraphicsView, self.ecgPlot)
        self.ecgGraphicsView.hide()
        self.ecgPlot.setBackground('black')
        self.ecgPlot.showGrid(x=True, y=True)
        self.ecgPlot.setLabel('left', 'Amplitude (mV)')
        self.ecgPlot.setLabel('bottom', 'Time (s)')
        
        # Initialize curve for ECG data
        self.ecg_curve = self.ecgPlot.plot(pen=pg.mkPen('b', width=2))
        
        # Data storage
        self.ecg_data = None
        self.sampling_rate = 250  # Default sampling rate
        self.data_index = 0
        self.is_monitoring = False

        self.afib_detected = False
        self.tachycardia_detected = False
        self.bradycardia_detected = False

        # Timer for real-time updates
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        
        # Alarm status
        self.alarm_active = False
        self.alarm_sound = None  
        
        # Connect buttons
        self.loadDataButton.clicked.connect(self.load_data)
        self.saveDataButton.clicked.connect(self.save_data)
        self.startMonitoringButton.clicked.connect(self.start_monitoring)
        self.stopMonitoringButton.clicked.connect(self.stop_monitoring)
        self.silenceAlarmButton.clicked.connect(self.silence_alarm)
        self.resetAlarmsButton.clicked.connect(self.reset_alarms)
        
        # Menu actions
        self.actionLoad_ECG_Data.triggered.connect(self.load_data)
        self.actionSave_ECG_Data.triggered.connect(self.save_data)
        self.actionExit.triggered.connect(self.close)
        self.actionAlarm_Thresholds.triggered.connect(self.open_threshold_settings)
        self.actionAbout.triggered.connect(self.show_about)
        
        # Initial UI setup
        self.stopMonitoringButton.setEnabled(False)
        self.silenceAlarmButton.setEnabled(False)
        
        # Store wfdb reference
        self.wfdb = None
        
        # Add history buffer for plotting
        self.display_buffer_size = 10 * 250  # 10 seconds at 250Hz (adjust based on your needs)
        self.ecg_display_buffer = np.zeros(self.display_buffer_size)
        
        # Initialize vital sign values
        self.bp_systolic = 120
        self.bp_diastolic = 80
        self.spo2 = 98
        self.temperature = 36.8
        self.resp_rate = 16
        
        # Initialize mockup data timer for vital signs
        self.vital_signs_timer = QtCore.QTimer()
        self.vital_signs_timer.timeout.connect(self.update_vital_signs)
        
        # Start vital signs updates when monitoring starts
        # We'll update these values with a bit of random variation to simulate real monitoring
    
    def set_wfdb(self, wfdb_module):
        self.wfdb = wfdb_module
        
    # All other methods remain the same, but update load_data to use self.wfdb
    def load_data(self):
        """Load ECG data from file"""
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open ECG Data", "", 
            "PhysioNet Files All Files (*);;(*.dat *.hea);;CSV Files (*.csv);;EDF Files (*.edf)", 
            options=options)
            
        if fileName:
            try:
                if fileName.endswith('.csv'):
                    self.ecg_data = np.loadtxt(fileName, delimiter=',')
                    self.sampling_rate = 250  # Assume 250Hz if not specified
                elif fileName.endswith('.edf'):
                    import pyedflib
                    
                    # Open the EDF file
                    f = pyedflib.EdfReader(fileName)
                    
                    # Get signal info
                    n_channels = f.signals_in_file
                    signal_labels = f.getSignalLabels()
                    
                    # Use the first channel if multiple exist (usually for ECG)
                    channel = 0
                    
                    # Get sampling frequency and data
                    self.sampling_rate = f.getSampleFrequency(channel)
                    self.ecg_data = f.readSignal(channel)
                    
                    f.close()
                else:
                    # Use WFDB to read PhysioNet format
                    if self.wfdb:
                        record = self.wfdb.rdrecord(fileName.replace('.dat', '').replace('.hea', ''))
                        self.ecg_data = record.p_signal[:, 0]  # First channel
                        self.sampling_rate = record.fs
                    else:
                        raise ImportError("wfdb module not available")
                
                # Display the first part of the data
                self.update_ecg_display(self.ecg_data[:1000])
                
                # Reset data index
                self.data_index = 0
                QMessageBox.information(self, "Data Loaded", f"Successfully loaded {len(self.ecg_data)} samples")
                
                # Enable start monitoring
                self.startMonitoringButton.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
    
    def save_data(self):
        """Save current ECG data to file"""
        if self.ecg_data is None:
            QMessageBox.warning(self, "Warning", "No data to save")
            return
            
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save ECG Data", "", 
            "CSV Files (*.csv);;All Files (*)", 
            options=options)
            
        if fileName:
            try:
                np.savetxt(fileName, self.ecg_data, delimiter=',')
                QMessageBox.information(self, "Success", "Data saved successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")
    
    def start_monitoring(self):
        """Start real-time ECG monitoring"""
        if self.ecg_data is None:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return
            
        self.is_monitoring = True
        self.startMonitoringButton.setEnabled(False)
        self.stopMonitoringButton.setEnabled(True)
        self.loadDataButton.setEnabled(False)
        
        # Start the timers
        self.update_timer.start(40)  # ~25 fps update rate for ECG
        self.vital_signs_timer.start(2000)  # Update vital signs every 2 seconds
    
    def stop_monitoring(self):
        """Stop real-time ECG monitoring"""
        self.is_monitoring = False
        self.update_timer.stop()
        self.vital_signs_timer.stop()
        
        self.startMonitoringButton.setEnabled(True)
        self.stopMonitoringButton.setEnabled(False)
        self.loadDataButton.setEnabled(True)
    
    def update_plot(self):
        """Update the ECG plot with new data"""
        if self.ecg_data is None or not self.is_monitoring:
            return
            
        # Calculate samples to show based on sampling rate and timer interval
        # For real-time playback: (sampling_rate * timer_interval_in_seconds)
        # 40ms timer interval = 0.04 seconds
        samples_per_update = int(self.sampling_rate * 0.04)  # Real-time playback
        
        # Get chunk of data to display
        chunk_size = min(samples_per_update, len(self.ecg_data) - self.data_index)
        
        if chunk_size <= 0:
            self.data_index = 0  # Loop back to beginning
            chunk_size = min(samples_per_update, len(self.ecg_data))
        
        # Get data chunk
        data_chunk = self.ecg_data[self.data_index:self.data_index + chunk_size]
        self.data_index += chunk_size
        
        # Update display
        self.update_ecg_display(data_chunk)
        
        # Take the last 2 seconds of ECG for analysis (2 * sampling_rate = 2000 samples)
        analysis_window_size = int(15 * self.sampling_rate)
        if self.data_index >= analysis_window_size:
            analysis_data = self.ecg_data[self.data_index - analysis_window_size:self.data_index]
        else:
            analysis_data = self.ecg_data[:self.data_index]

        self.analyze_ecg(analysis_data)

    def update_ecg_display(self, new_data):
        """Update the ECG plot with new data"""
        # Shift the buffer left
        data_len = len(new_data)
        if data_len > 0:
            # Shift buffer left and add new data at the end
            self.ecg_display_buffer[:-data_len] = self.ecg_display_buffer[data_len:]
            self.ecg_display_buffer[-data_len:] = new_data
            
            # Generate time values (x-axis)
            time_values = np.arange(self.display_buffer_size) / self.sampling_rate
            
            # Update plot
            self.ecg_curve.setData(time_values, self.ecg_display_buffer)
    
    def analyze_ecg(self, data):
        """Analyze ECG data for arrhythmias"""
        # Calculate heart rate
        heart_rate = self.detector.detect_heart_rate(data, fs=self.sampling_rate)
        if heart_rate > 0:
            self.last_valid_hr = heart_rate
        else:
            if self.last_valid_hr is not None:
                heart_rate = self.last_valid_hr
            else:
                heart_rate = 0  # fallback if no valid HR yet

        self.heartRateLabel.setText(f"{heart_rate:.1f}")
        
        # Check for arrhythmias
        if self.detector.detect_tachycardia(heart_rate):
            self.tachycardia_detected = True

        elif self.detector.detect_bradycardia(heart_rate):
            self.bradycardia_detected = True

        elif self.detector.detect_afib(data, fs=self.sampling_rate):
            self.afib_detected = True

        # Update status labels
        self.update_arrhythmia_status(self.tachycardia_detected , self.bradycardia_detected , self.afib_detected)
        
        # Trigger alarm if needed
        if (self.tachycardia_detected or self.bradycardia_detected or self.afib_detected):
            print(heart_rate)
            self.trigger_alarm()
    
    def update_arrhythmia_status(self, tachycardia_detected , bradycardia_detected  , afib_detected):
        """Update the arrhythmia status labels"""
        # Update tachycardia status
        if tachycardia_detected:
            self.tachStatusLabel.setText("DETECTED")
            self.tachStatusLabel.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.tachStatusLabel.setText("Normal")
            self.tachStatusLabel.setStyleSheet("color: green;")

        # Update bradycardia status
        if bradycardia_detected:
            self.bradStatusLabel.setText("DETECTED")
            self.bradStatusLabel.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.bradStatusLabel.setText("Normal")
            self.bradStatusLabel.setStyleSheet("color: green;")

        # Update afib status
        if afib_detected:
            self.afibStatusLabel.setText("DETECTED")
            self.afibStatusLabel.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.afibStatusLabel.setText("Normal")
            self.afibStatusLabel.setStyleSheet("color: green;")

    def trigger_alarm(self):
        """Trigger the alarm for arrhythmia detection"""
        if not self.alarm_active:
            self.alarm_active = True
            self.silenceAlarmButton.setEnabled(True)

            if self.alarmEnabledCheckBox.isChecked():
                self.alarm_sound = QSound("alarm.wav")
                self.alarm_sound.play()
                
            # Flash the screen red
            self.flash_timer = QtCore.QTimer()
            self.flash_timer.timeout.connect(self.flash_alarm)
            self.flash_timer.start(500)  # Flash every 500ms
    
    def flash_alarm(self):
        """Flash the background of the alarm frame to indicate alarm"""
        if self.alarmFrame.property("alarm_on"):
            self.alarmFrame.setStyleSheet("background-color: white;")
            self.alarmFrame.setProperty("alarm_on", False)
        else:
            self.alarmFrame.setStyleSheet("background-color: red;")
            self.alarmFrame.setProperty("alarm_on", True)
    
    def silence_alarm(self):
        """Silence the current alarm"""
        if self.alarm_active:
            self.alarm_active = False
            if hasattr(self, 'flash_timer'):
                self.flash_timer.stop()
            self.alarmFrame.setStyleSheet("")
            self.silenceAlarmButton.setEnabled(False)
            if self.alarm_sound:
               self.alarm_sound.stop()
    
    def reset_alarms(self):
        """Reset all alarms to normal state"""
        self.silence_alarm()
        for label in [self.tachStatusLabel, self.bradStatusLabel, self.afibStatusLabel]:
                label.setText("Normal")
                label.setStyleSheet("color: green;")

        self.tachycardia_detected = False
        self.bradycardia_detected = False
        self.afib_detected = False

    def open_threshold_settings(self):
        """Open dialog to adjust alarm thresholds"""
        # This would be implemented as a dialog to adjust threshold values
        QMessageBox.information(self, "Not Implemented", "Threshold settings dialog not implemented in this demo")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", "ECG Monitoring System\nVersion 1.0\nDeveloped for medical professionals")
    
    def update_vital_signs(self):
        """Update vital signs with realistic variations"""
        import random
        
        # Update blood pressure with small variations
        self.bp_systolic = max(90, min(180, self.bp_systolic + random.uniform(-3, 3)))
        self.bp_diastolic = max(50, min(110, self.bp_diastolic + random.uniform(-2, 2)))
        self.bpLabel.setText(f"{int(self.bp_systolic)}/{int(self.bp_diastolic)}")
        
        # Update SpO2 with small variations (healthy range is 95-100%)
        self.spo2 = max(80, min(100, self.spo2 + random.uniform(-1, 1)))
        self.spo2Label.setText(f"{int(self.spo2)}")
        
        # Update temperature with tiny variations
        self.temperature = max(35, min(39, self.temperature + random.uniform(-0.1, 0.1)))
        self.tempLabel.setText(f"{self.temperature:.1f}")
        
        # Update respiratory rate with small variations
        self.resp_rate = max(10, min(25, self.resp_rate + random.uniform(-1, 1)))
        self.respRateLabel.setText(f"{int(self.resp_rate)}")
        
        # Update colors based on values
        self.update_vital_sign_colors()
    
    def update_vital_sign_colors(self):
        """Update colors of vital signs based on their values"""
        # Heart rate colors
        heart_rate = float(self.heartRateLabel.text()) if self.heartRateLabel.text() != "--" else 0
        if heart_rate < 50 or heart_rate > 100:
            self.heartRateLabel.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.heartRateLabel.setStyleSheet("color: green; font-weight: bold;")
        
        # Blood pressure colors
        if self.bp_systolic > 140 or self.bp_diastolic > 90 or self.bp_systolic < 90 or self.bp_diastolic < 60:
            self.bpLabel.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.bpLabel.setStyleSheet("color: green; font-weight: bold;")
        
        # SpO2 colors
        if self.spo2 < 95:
            self.spo2Label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.spo2Label.setStyleSheet("color: green; font-weight: bold;")
        
        # Temperature colors
        if self.temperature > 37.5 or self.temperature < 36.0:
            self.tempLabel.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.tempLabel.setStyleSheet("color: green; font-weight: bold;")
        
        # Respiratory rate colors
        if self.resp_rate < 12 or self.resp_rate > 20:
            self.respRateLabel.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.respRateLabel.setStyleSheet("color: green; font-weight: bold;")


def main():
    # Create QApplication first
    app = QtWidgets.QApplication(sys.argv)
    
    # Now import modules that might create widgets
    import pyqtgraph as pg
    import wfdb

    # Create the window and pass pyqtgraph module
    window = ECGMonitoringSystem(pg)
    
    # Set wfdb module
    window.set_wfdb(wfdb)
    
    # Show window
    window.show()
    
    # Execute app
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()