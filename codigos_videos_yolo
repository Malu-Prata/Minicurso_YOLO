# ---- DETECÇÃO DE OBJETOS EM VÍDEOS COM YOLOv8 ---- 

# -- Instala a biblioteca ultralytics que o YOLOv8 usa na detecção de objetos
!pip install ultralytics                

# --- Importa as bibliotecas necessárias ---
from google.colab import files                 # Para fazer upload de arquivos do computador para o Colab
from ultralytics import YOLO                   # Para usar os modelos YOLOv8 (da biblioteca 'ultralytics')
from IPython.display import Video, display     # Para exibir o vídeo com as detecções direto no notebook
import os                                      # Para navegar entre pastas e arquivos
import shutil                                  # Para copiar arquivos de uma pasta para outra
import cv2                                     # Para recodificar o vídeo

# --- Faz o upload do vídeo que vai ser processado ---
print("Envie um arquivo de vídeo (formato .mp4, .avi, etc.)")
uploaded = files.upload()                      # Abre a janela para o usuário enviar um ou mais arquivos

# Mostra os arquivos enviados
print("Arquivos enviados:")
for filename in uploaded.keys():
    print(f"- {filename}")

# --- Pede o nome do vídeo enviado pra ser processado ---
# O nome deve ser exatamente igual ao que foi enviado acima
video_name = input("Digite o nome exato do vídeo enviado (ex: video.mp4): ")

# --- Verifica se o vídeo foi realmente enviado ---
if video_name not in uploaded:
    print("Erro: vídeo não encontrado. Verifique o nome digitado.")
else:
    # --- Carrega o modelo YOLOv8 ---
    # Aqui usamos o modelo pequeno 'yolov8n.pt' para rapidez (pode trocar por outro: s, m, l, x)

    model = YOLO('yolov8n.pt')

    # --- Executa a detecção no vídeo escolhido ---
    results = model.predict(
        source=video_name,     # Caminho do vídeo que foi enviado
        save=True,             # Salva o vídeo com as caixas de detecção desenhadas
        save_txt=False,        # Não salva os resultados em arquivos .txt (pode ativar se quiser)
        conf=0.25              # Limite de confiança (de 0 a 1); abaixo disso, ignora as detecções
    )

    # --- Procura o vídeo de saída salvo pelo YOLO ---
    output_dir = 'runs/detect/predict'         # Pasta padrão onde o YOLO salva os resultados
    output_video = None

    # Percorre os arquivos na pasta de saída
    for file in os.listdir(output_dir):
      if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # aceita outros formatos
        output_video = os.path.join(output_dir, file)
        break

# --- Exibir vídeo no colab ---
if output_video:
    # Recodifica o vídeo usando FFmpeg para máxima compatibilidade
    output_path = './output_detected.mp4'

    print("Recodificando vídeo para compatibilidade com navegador...")

    # Usa FFmpeg para recodificar com configurações específicas para web
    import subprocess

    cmd = [
        'ffmpeg', '-i', output_video, '-y',  # -y para sobrescrever
        '-c:v', 'libx264',           # Codec de vídeo H.264
        '-c:a', 'aac',               # Codec de áudio AAC
        '-movflags', '+faststart',   # Otimiza para streaming web
        '-pix_fmt', 'yuv420p',       # Formato de pixel compatível
        '-preset', 'fast',           # Preset de codificação rápida
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Recodificação concluída com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg falhou, tentando com OpenCV...")
        # Fallback para OpenCV se FFmpeg falhar
        cap = cv2.VideoCapture(output_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec mais compatível
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        print("Recodificação com OpenCV concluída!")

    print(f"\nVídeo com detecções salvo como: output_detected.mp4")

    # Exibe o vídeo (método que funcionou)
    print("Exibindo vídeo...")

    # Verifica se o arquivo foi criado corretamente
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Arquivo criado: {os.path.getsize(output_path)} bytes")

        # Exibe o vídeo com embed
        display(Video(output_path, embed=True, width=640, height=480, html_attributes="controls"))
        print("Vídeo exibido com sucesso!")
    else:
        print("Erro: arquivo de vídeo não foi criado corretamente.")

    # Link simples para download
    print("\nClique abaixo para baixar o vídeo:")
    from google.colab import files as colab_files
    from IPython.display import HTML
    
    download_html = f'''
    <a href="javascript:void(0)" onclick="google.colab.kernel.invokeFunction('download_video', [], {{}})" 
       style="color: #1976d2; text-decoration: underline;">
        Fazer o download do vídeo
    </a>
    '''
    display(HTML(download_html))
    
    # Função JavaScript para download
    from google.colab import output
    output.register_callback('download_video', lambda: colab_files.download(output_path))
else:
    print("Erro: vídeo de saída não foi encontrado.")
