import os
import subprocess

def run_command(command):
    """Executa um comando no terminal e exibe a saída em tempo real."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end='')
    for line in process.stderr:
        print(line, end='')
    process.wait()
    return process.returncode

def main():
    print("Instalando dependências...")
    run_command("pip install torch torchvision pretty_midi")

    print("Clonando o repositório do LakhNES...")
    if not os.path.exists("LakhNES"):
        run_command("git clone https://github.com/chrisdonahue/LakhNES.git")
    os.chdir("LakhNES")

    print("Criando ambiente virtual...")
    run_command("python -m venv LakhNES-model")
    run_command(r"LakhNES-model\Scripts\activate && pip install torch torchvision")

    print("Baixando modelo pré-treinado...")
    model_path = "model/LakhNES.pth"
    if not os.path.exists(model_path):
        run_command(f"wget -O {model_path} https://drive.google.com/uc?export=download&id=1ND27trP3pTAl6eAk5QiYE9JjLOivqGsd")

    print("Gerando música...")
    run_command("python generate.py model/LakhNES.pth --out_dir ./generated --num 1")

    print("Convertendo para áudio...")
    run_command("python data/synth_client.py ./generated/0.tx1.txt ./generated/0.tx1.wav")

    print("Execução concluída! Confira o arquivo gerado em ./generated/0.tx1.wav")

if __name__ == "__main__":
    main()
