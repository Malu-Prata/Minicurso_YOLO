!pip install ultralytics

# --- IMPORTAÇÕES ---
from google.colab import files
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- UPLOAD DAS IMAGENS ---
uploaded = files.upload()

# --- LISTA OS ARQUIVOS ENVIADOS ---
print("Arquivos enviados:")
for filename in uploaded.keys():
    print(f"- {filename}")

# --- ESCOLHE A IMAGEM PELO NOME ---
img_name = input("Digite o nome exato da imagem que deseja processar com pose: ")

# --- VERIFICA SE A IMAGEM EXISTE ---
if img_name not in uploaded:
    print("Erro: imagem não encontrada! Confira o nome e tente novamente.")
else:
    # --- CARREGA O MODELO DE ESTIMATIVA DE POSE ---
    model = YOLO('yolov8n-pose.pt')  # Você pode trocar por yolov8s-pose.pt, yolov8m-pose.pt, etc.

    # --- EXECUTA A ESTIMATIVA DE POSE ---
    results = model(img_name)

    # --- MOSTRA A IMAGEM ORIGINAL ---
    imagem = mpimg.imread(img_name)
    plt.imshow(imagem)
    plt.title(f"Imagem Original: {img_name}")
    plt.axis('off')
    plt.show()

    # --- MOSTRA A IMAGEM COM AS POSES ESTIMADAS ---
    plt.imshow(results[0].plot())
    plt.title(f"Estimativa de Pose: {img_name}")
    plt.axis('off')
    plt.show()
