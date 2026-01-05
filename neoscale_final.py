import cv2
from ultralytics import YOLO
import time
import csv
import os
import sys
import RPi.GPIO as GPIO

class HX711:
    def __init__(self, dout, pd_sck, gain=128):
        self.PD_SCK = pd_sck
        self.DOUT = dout
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.PD_SCK, GPIO.OUT)
        GPIO.setup(self.DOUT, GPIO.IN)
        self.GAIN = 0
        self.REFERENCE_UNIT = 1  
        self.OFFSET = 1
        self.set_gain(gain)

    def is_ready(self):
        return GPIO.input(self.DOUT) == 0

    def set_gain(self, gain):
        if gain == 128: self.GAIN = 1
        elif gain == 64: self.GAIN = 3
        elif gain == 32: self.GAIN = 2
        GPIO.output(self.PD_SCK, False)
        self.read()

    def read(self):
        while not self.is_ready(): pass
        dataValue = 0x00
        for i in range(24):
            GPIO.output(self.PD_SCK, True)
            dataValue = dataValue << 1
            GPIO.output(self.PD_SCK, False)
            if GPIO.input(self.DOUT): dataValue += 1
        
        # Set gain for next reading
        for i in range(self.GAIN):
            GPIO.output(self.PD_SCK, True)
            GPIO.output(self.PD_SCK, False)
        
        if dataValue & 0x800000: dataValue -= 0x1000000 
        return dataValue

    def read_average(self, times=3):
        sum = 0
        for i in range(times): sum += self.read()
        return sum / times

    def get_value(self, times=3):
        return self.read_average(times) - self.OFFSET

    def get_weight(self, times=3):
        value = self.get_value(times)
        value = value / self.REFERENCE_UNIT
        return value

    def tare(self, times=15):
        sum = self.read_average(times)
        self.set_offset(sum)

    def set_offset(self, offset):
        self.OFFSET = offset

    def set_reference_unit(self, reference_unit):
        self.REFERENCE_UNIT = reference_unit

    def power_down(self):
        GPIO.output(self.PD_SCK, False)
        GPIO.output(self.PD_SCK, True)
        time.sleep(0.0001)

    def power_up(self):
        GPIO.output(self.PD_SCK, False)

MODEL_PATH = "best.pt"       
DATABASE_FILE = "harga.csv"  
PIN_DT = 5                   
PIN_SCK = 6                  
REF_UNIT = 99.3              

def load_harga_dari_csv(filename):
    database = {}
    if not os.path.exists(filename):
        print("WARNING: File harga.csv tidak ditemukan!")
        return database
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    try:
                        nama = row[0].strip()
                        harga = int(row[1].strip())
                        database[nama] = harga
                    except: pass
        print(f"Database Harga Terload: {len(database)} barang.")
    except Exception as e:
        print(f"Error CSV: {e}")
    return database

def main():
    print("=== MEMULAI NEOSCALE SYSTEM ===")
    
    # 1. SETUP SENSOR BERAT
    print("[1/3] Menghubungkan Sensor Berat...")
    try:
        hx = HX711(PIN_DT, PIN_SCK)
        hx.set_reference_unit(REF_UNIT)
        hx.tare()
        print("      -> Sensor Siap! (Timbangan di-Nol-kan)")
    except Exception as e:
        print(f"      -> ERROR SENSOR: {e}")
        sys.exit()

    # 2. SETUP KAMERA & AI
    print("[2/3] Menghubungkan Kamera & AI...")
    try:
        model = YOLO(MODEL_PATH)
        
        # Inisialisasi Kamera
        cap = cv2.VideoCapture(0)
        
        # Setting Resolusi Rendah (Agar Lancar di VNC/Pi 5)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Kurangi buffer biar realtime
        
        if not cap.isOpened():
            print("      -> WARNING: Kamera tidak merespon index 0.")
        else:
            print("      -> Kamera Terhubung!")
            
    except Exception as e:
        print(f"      -> ERROR KAMERA/AI: {e}")
        sys.exit()

    # 3. SETUP DATABASE
    print("[3/3] Membaca Database Harga...")
    HARGA_PER_GRAM = load_harga_dari_csv(DATABASE_FILE)

    print("\nSISTEM AKTIF! Tekan 'q' untuk keluar.")
    print("Letakkan barang untuk mulai...")

    # --- LOOPING UTAMA ---
    while True:
        # A. BACA KAMERA
        ret, frame = cap.read()
        
      
        if not ret or frame is None:
            frame = cv2.imread("tes.jpg") 
            if frame is None:
                
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "KAMERA ERROR", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # B. BACA BERAT (Ambil rata-rata 3x biar cepat tapi stabil)
        berat_raw = hx.get_weight(3)
        berat_fix = int(max(0, berat_raw)) # Tidak boleh minus

        # C. LOGIKA DETEKSI 
        detected_name = "..."
        harga_satuan = 0
        box_color = (100, 100, 100) 
    
        if berat_fix > 10:
            
            results = model(frame, verbose=False, conf=0.5)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    
                    if label in HARGA_PER_GRAM:
                        detected_name = label
                        harga_satuan = HARGA_PER_GRAM[label]
                        box_color = (0, 255, 0)
                    else:
                        detected_name = label + " (?)"
                        box_color = (0, 255, 255)
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    break 

        # D. HITUNG TOTAL
        total_bayar = berat_fix * harga_satuan

        # E. TAMPILAN UI KASIR
        cv2.rectangle(frame, (0, 0), (640, 120), (0, 0, 0), -1)
        
        cv2.putText(frame, f"ITEM : {detected_name}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if berat_fix > 10:
            info_berat = f"Berat: {berat_fix}g  x  Rp {harga_satuan}"
            cv2.putText(frame, info_berat, (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.putText(frame, "TOTAL:", (15, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            str_total = f"Rp {total_bayar:,}"
            cv2.putText(frame, str_total, (100, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SILAKAN LETAKKAN BARANG", (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)

        # F. TAMPILKAN
        cv2.imshow("Neoscale Final System", frame)

        # Reset Sensor Power
        hx.power_down()
        hx.power_up()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

if __name__ == "__main__":
    main()
