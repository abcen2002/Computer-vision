import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time

class GUI:
    def __init__(self,root):
        self.root = root
        self.root.title("Hệ thống nhận diện biển báo")
        
        # Cấu hình kích thước màn hình và căn giữa
        app_width = 1100
        app_height = 650
        
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        x = (screen_width - app_width) // 2
        y = (screen_height - app_height) // 2
        
        self.root.geometry(f"{app_width}x{app_height}+{x}+{y}")
        
        # Biến trạng thái của class GUI
        self.dang_chay = False
        self.cap = None
        self.vi_tri_video_hien_tai = 0 

        # Giao diện
        # Khung Trái
        self.khung_trai = tk.Frame(root, bg="whitesmoke", width=320)
        self.khung_trai.pack(side="left", fill="y")
        self.khung_trai.pack_propagate(False)
        
        # Khung Phải
        self.khung_phai = tk.Frame(root, bg="#e0e0e0")
        self.khung_phai.pack(side="right", fill="both", expand=True)

        self.tao_widgets_ben_trai()
        self.tao_man_hinh_ben_phai()

    def tao_widgets_ben_trai(self):
        # Chọn Model
        tk.Label(self.khung_trai, text="Chọn Model (.pt)", bg="whitesmoke", font=("Arial", 10, "bold")).pack(pady=(15,5))
        self.entry_model = tk.Entry(self.khung_trai, width=35)
        self.entry_model.pack(padx=5)
        self.entry_model.insert(0, "yolo11_1/bestYolo.pt")
        tk.Button(self.khung_trai, text="Chọn Model", command=self.chon_model).pack(pady=5)

        # Chọn File
        tk.Label(self.khung_trai, text="Chọn Ảnh/Video", bg="whitesmoke", font=("Arial", 10, "bold")).pack(pady=(15,5))
        self.entry_media = tk.Entry(self.khung_trai, width=35)
        self.entry_media.pack(padx=5)
        self.entry_media.insert(0, "testImage/image_018.jpg")
        tk.Button(self.khung_trai, text="Chọn File", command=self.chon_media).pack(pady=5)

        # Tùy chọn Tốc độ
        tk.Label(self.khung_trai, text="Tùy chọn bỏ qua khung hình", bg="whitesmoke", font=("Arial", 10, "bold")).pack(pady=(20,0))
        self.slider_toc_do = tk.Scale(self.khung_trai, from_=0, to=10, orient="horizontal", length=250, bg="whitesmoke")
        self.slider_toc_do.set(0)
        self.slider_toc_do.pack(pady=5)
        tk.Label(self.khung_trai, text="(Kéo lên nếu muốn tua)", bg="whitesmoke", font=("Arial", 8, "italic")).pack()

        # Nút Lệnh
        tk.Label(self.khung_trai, text="----------------", bg="whitesmoke").pack(pady=(20,10))
        
        self.btn_start = tk.Button(self.khung_trai, text="CHẠY / TIẾP TỤC", bg="#28a745", fg="white", 
                                   font=("Arial", 12, "bold"), width=18, command=self.bat_dau_xu_ly)
        self.btn_start.pack(pady=5)

        self.btn_stop = tk.Button(self.khung_trai, text="TẠM DỪNG", bg="#ffc107", fg="black", 
                                  font=("Arial", 12, "bold"), width=18, command=self.dung_xu_ly, state="disabled")
        self.btn_stop.pack(pady=5)

        self.btn_reset = tk.Button(self.khung_trai, text="CHẠY LẠI TỪ ĐẦU", bg="#dc3545", fg="white", 
                                  font=("Arial", 12, "bold"), width=18, command=self.reset_video)
        self.btn_reset.pack(pady=5)
        
        # status of GUI
        self.lbl_status = tk.Label(self.khung_trai, text="Sẵn sàng", fg="blue", bg="whitesmoke")
        self.lbl_status.pack(side="bottom", pady=20)

    def tao_man_hinh_ben_phai(self):
        # Màn hình bên phải
        self.lbl_hien_thi = tk.Label(self.khung_phai, text="Màn hình chính", 
                                     bg="white", fg="black", font=("Arial", 14), bd=2, relief="ridge")
        
        self.lbl_hien_thi.pack(expand=True, fill="both", padx=20, pady=20)

    # Xử lý sự kiện
    def chon_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if path:
            self.entry_model.delete(0, tk.END)
            self.entry_model.insert(0, path)

    def chon_media(self):
        path = filedialog.askopenfilename(title="Chọn file",
                                          filetypes=[("Media files", "*.jpg *.png *.mp4 *.avi *.mkv")])
        if path:
            self.entry_media.delete(0, tk.END)
            self.entry_media.insert(0, path)
            self.vi_tri_video_hien_tai = 0
            self.lbl_status.config(text="Đã chọn file mới", fg="blue")

    def reset_video(self):
        self.dung_xu_ly() 
        self.vi_tri_video_hien_tai = 0 
        self.lbl_status.config(text="Đã tua về đầu", fg="blue")

    def bat_dau_xu_ly(self):
        model_path = self.entry_model.get()
        media_path = self.entry_media.get()
        if not model_path or not media_path:
            messagebox.showwarning("Cảnh báo", "Hãy chọn đủ Model và File!")
            return
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_status.config(text="Đang xử lý...", fg="orange")
        self.dang_chay = True
        threading.Thread(target=self.chay_logic_yolo, args=(model_path, media_path), daemon=True).start()

    def dung_xu_ly(self):
        self.dang_chay = False
        if self.cap:
            self.cap.release()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        if self.vi_tri_video_hien_tai > 0:
            self.lbl_status.config(text=f"Tạm dừng tại Frame {int(self.vi_tri_video_hien_tai)}", fg="brown")
        else:
            self.lbl_status.config(text="Đã dừng", fg="red")

    def chay_logic_yolo(self, model_path, media_path):
        try:
            model = YOLO(model_path)
            is_image = media_path.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp'))

            #image
            if is_image:
                results = model(media_path)
                frame = results[0].plot()
                self.hien_thi_frame(frame)
                self.root.after(0, lambda: self.dung_xu_ly())
            #video
            else:
                self.cap = cv2.VideoCapture(media_path)
                if self.vi_tri_video_hien_tai > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.vi_tri_video_hien_tai)

                frame_count = 0 
                TARGET_WIDTH = 640

                while self.dang_chay and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret: 
                        self.vi_tri_video_hien_tai = 0
                        break
                    
                    self.vi_tri_video_hien_tai = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    frame_count += 1
                    
                    skip_val = self.slider_toc_do.get()
                    if skip_val > 0 and frame_count % (skip_val + 1) != 0:
                        continue 

                    height, width = frame.shape[:2]
                    if width > TARGET_WIDTH:
                        scale = TARGET_WIDTH / width
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (TARGET_WIDTH, new_height))

                    results = model.predict(frame, conf=0.5, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    self.hien_thi_frame(annotated_frame)
                    
                    if skip_val == 0:
                        time.sleep(0.01)
                    
                self.cap.release()
                self.root.after(0, lambda: self.dung_xu_ly())

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Sự cố: {e}"))
            self.root.after(0, self.dung_xu_ly)

    def hien_thi_frame(self, cv2_img):
        w_view = self.khung_phai.winfo_width() - 40
        h_view = self.khung_phai.winfo_height() - 40
        
        if w_view < 10: w_view = 640 
        if h_view < 10: h_view = 640

        cv2_img = cv2.resize(cv2_img, (w_view, h_view))
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.root.after(0, lambda: self.cap_nhat_label(img_tk))

    def cap_nhat_label(self, img_tk):
        if self.dang_chay or self.cap is None:
            self.lbl_hien_thi.config(image=img_tk, text="")
            self.lbl_hien_thi.image = img_tk

if __name__ == "__main__":
    manHinh = tk.Tk()
    app = GUI(manHinh)
    manHinh.mainloop()