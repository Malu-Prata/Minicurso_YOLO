!pip install ultralytics

from ultralytics import YOLO
import cv2
from google.colab import files
import numpy as np
from IPython.display import HTML
from base64 import b64encode
import os

# Instala ultralytics
!pip install -q ultralytics

# Upload do vídeo
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# Carrega modelo YOLOv8 pose
model = YOLO('yolov8n-pose.pt')

# Função melhorada para detectar queda
def check_fall(keypoints):
    """
    Detecta queda baseado na posição dos keypoints
    Retorna: (is_fall, fall_info)
    """
    # Filtra keypoints com confiança > 0.3
    valid_kp = [(kp[0], kp[1]) for kp in keypoints if kp[2] > 0.3]
    if len(valid_kp) < 5:  # Precisa de pelo menos 5 pontos válidos
        return False, "Poucos pontos detectados"
    
    y_coords = [kp[1] for kp in valid_kp]
    x_coords = [kp[0] for kp in valid_kp]
    
    # Calcula altura da pessoa (diferença entre ponto mais alto e mais baixo)
    height_diff = max(y_coords) - min(y_coords)
    
    # Calcula largura da pessoa
    width_diff = max(x_coords) - min(x_coords)
    
    # Critérios para queda:
    # 1. Altura muito pequena (pessoa "deitada")
    # 2. Largura maior que altura (pessoa horizontal)
    is_fall = height_diff < 80 or width_diff > height_diff * 1.5
    
    info = f"H:{height_diff:.0f} W:{width_diff:.0f}"
    
    return is_fall, info

# Processa o vídeo
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Saída de vídeo com codec H.264 (mais compatível)
output_path = 'saida_queda_colab.mp4'
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # ou tente 'H264'
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

print(f"Processando vídeo: {w}x{h} @ {fps} fps")

frame_count = 0
fall_detected_frames = []  # Lista para armazenar frames com queda

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 30 == 0:  # Mostra progresso a cada 30 frames
        print(f"Processando frame {frame_count}")
    
    results = model.predict(frame, conf=0.3, iou=0.3, device='cpu')
    res = results[0]
    annotated = res.plot()
    
    # Variável para controlar se há queda neste frame
    fall_in_frame = False
    
    if res.keypoints is not None and res.keypoints.data is not None:
        for i, kp in enumerate(res.keypoints.data):
            kp_np = kp.cpu().numpy()
            is_fall, fall_info = check_fall(kp_np)
            
            if is_fall:
                fall_in_frame = True
                fall_detected_frames.append(frame_count)
                
                # Desenha retângulo vermelho ao redor da tela
                cv2.rectangle(annotated, (0, 0), (w-1, h-1), (0, 0, 255), 8)
                
                # Mensagem grande de alerta
                cv2.putText(annotated, '⚠️ QUEDA DETECTADA! ⚠️', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # Informações técnicas
                cv2.putText(annotated, f'Pessoa {i+1}: {fall_info}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Timestamp
                timestamp = f'Frame: {frame_count} | Tempo: {frame_count/fps:.1f}s'
                cv2.putText(annotated, timestamp, (10, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Adiciona contador de frames no canto superior direito
    cv2.putText(annotated, f'Frame: {frame_count}', (w-200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(annotated)

cap.release()
out.release()

print("Processamento concluído!")
print(f"Total de frames processados: {frame_count}")

# Relatório de quedas detectadas
if fall_detected_frames:
    print(f"\n🚨 QUEDAS DETECTADAS em {len(fall_detected_frames)} frames:")
    unique_falls = []
    current_fall = [fall_detected_frames[0]]
    
    # Agrupa frames consecutivos em "eventos de queda"
    for i in range(1, len(fall_detected_frames)):
        if fall_detected_frames[i] - fall_detected_frames[i-1] <= 5:  # Frames próximos = mesmo evento
            current_fall.append(fall_detected_frames[i])
        else:
            unique_falls.append(current_fall)
            current_fall = [fall_detected_frames[i]]
    unique_falls.append(current_fall)
    
    for i, fall_event in enumerate(unique_falls):
        start_time = fall_event[0] / fps
        end_time = fall_event[-1] / fps
        print(f"  Evento {i+1}: {start_time:.1f}s - {end_time:.1f}s (frames {fall_event[0]}-{fall_event[-1]})")
else:
    print("\n✅ Nenhuma queda detectada no vídeo")

# Verifica se o arquivo foi criado corretamente
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path)
    print(f"Arquivo criado: {output_path} ({file_size} bytes)")
    
    # Método 1: Reprodução direta com HTML5
    try:
        with open(output_path, 'rb') as f:
            mp4_data = f.read()
        data_url = "data:video/mp4;base64," + b64encode(mp4_data).decode()
        
        display(HTML(f"""
        <video width="640" height="480" controls>
            <source src="{data_url}" type="video/mp4">
            Seu navegador não suporta vídeo HTML5.
        </video>
        """))
    except Exception as e:
        print(f"Erro na reprodução HTML: {e}")
        
        # Método 2: Download do arquivo (alternativa)
        print("Baixando arquivo para visualização local...")
        files.download(output_path)
        
else:
    print("Erro: Arquivo de saída não foi criado!")

# Método 3: Alternativa usando FFmpeg para garantir compatibilidade
print("\n--- Tentativa com FFmpeg (mais compatível) ---")
try:
    # Reprocessa com FFmpeg para garantir compatibilidade
    output_path_ffmpeg = 'saida_queda_final.mp4'
    
    # Usa FFmpeg para converter
    !ffmpeg -i {output_path} -c:v libx264 -c:a aac -strict experimental -b:a 128k -movflags +faststart {output_path_ffmpeg} -y
    
    # Exibe a versão FFmpeg
    with open(output_path_ffmpeg, 'rb') as f:
        mp4_data = f.read()
    data_url = "data:video/mp4;base64," + b64encode(mp4_data).decode()
    
    display(HTML(f"""
    <div style="text-align: center;">
        <h3>Vídeo com Detecção de Queda</h3>
        <video width="640" height="480" controls>
            <source src="{data_url}" type="video/mp4">
            Seu navegador não suporta vídeo HTML5.
        </video>
    </div>
    """))
    
except Exception as e:
    print(f"Erro com FFmpeg: {e}")
    print("Baixando arquivo original...")
    files.download(output_path)
