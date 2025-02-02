import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
from sklearn.cluster import KMeans
from mido import Message, MidiFile, MidiTrack
import logging
from typing import Tuple, List, Optional
import soundfile as sf
import os

MUSIC_STYLES = {
    'NES Original': {
        'pulse1_velocity': 100,
        'pulse2_velocity': 80,
        'triangle_velocity': 90,
        'noise_velocity': 70,
        'note_duration': {'pulse1': 240, 'pulse2': 240, 'triangle': 240, 'noise': 120}
    },
    'Rock': {
        'pulse1_velocity': 120,  # Guitarra principal mais forte
        'pulse2_velocity': 100,  # Guitarra base
        'triangle_velocity': 110,  # Baixo mais presente
        'noise_velocity': 100,  # Bateria mais forte
        'note_duration': {'pulse1': 200, 'pulse2': 200, 'triangle': 280, 'noise': 160}
    },
    'Pop': {
        'pulse1_velocity': 90,  # Melodia mais suave
        'pulse2_velocity': 85,  # Harmonia equilibrada
        'triangle_velocity': 85,  # Baixo moderado
        'noise_velocity': 80,  # Batida pop
        'note_duration': {'pulse1': 220, 'pulse2': 220, 'triangle': 220, 'noise': 140}
    },
    'Jazz': {
        'pulse1_velocity': 85,  # Melodia jazz
        'pulse2_velocity': 80,  # Acordes jazz
        'triangle_velocity': 90,  # Walking bass
        'noise_velocity': 60,  # Percussão suave
        'note_duration': {'pulse1': 180, 'pulse2': 200, 'triangle': 160, 'noise': 100}
    },
    'Eletrônico': {
        'pulse1_velocity': 110,  # Lead synth
        'pulse2_velocity': 95,   # Pad/ambiente
        'triangle_velocity': 100, # Bass synth
        'noise_velocity': 90,    # Beats eletrônicos
        'note_duration': {'pulse1': 160, 'pulse2': 300, 'triangle': 200, 'noise': 140}
    }
}

def freq_to_midi_note(frequency: float) -> int:
    """Converte frequência em Hz para nota MIDI."""
    if frequency <= 0:
        return 0
    return int(round(69 + 12 * np.log2(frequency / 440.0)))

def validate_audio_parameters(samples: np.ndarray, sr: int) -> bool:
    """Valida os parâmetros básicos do áudio."""
    if len(samples) == 0:
        raise ValueError("Amostra de áudio vazia")
    if sr not in [44100, 48000, 22050]:
        raise ValueError(f"Taxa de amostragem {sr} não suportada")
    if len(samples.shape) > 1:
        raise ValueError("Áudio deve ser mono")
    return True

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Carrega e pré-processa o arquivo de áudio."""
    try:
        # Carregar áudio
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(44100)
        
        # Converter para numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = librosa.util.normalize(samples)
        
        validate_audio_parameters(samples, audio.frame_rate)
        return samples, audio.frame_rate
    
    except Exception as e:
        logging.error(f"Erro ao carregar arquivo {file_path}: {str(e)}")
        raise

def detect_note_onsets(samples: np.ndarray, sr: int) -> np.ndarray:
    """Detecta os pontos de início das notas."""
    onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        wait=1,  # Mínimo de frames entre onsets
        pre_max=3,  # Janela de look-ahead
        post_max=3,  # Janela de look-behind
        pre_avg=3,
        post_avg=5,
        delta=0.1,  # Sensibilidade
        normalize=True
    )
    return librosa.frames_to_time(onset_frames, sr=sr)

def quantize_timing(onset_times: np.ndarray, bpm: float) -> np.ndarray:
    """Quantiza o timing das notas para o grid do NES."""
    # NES usa 24 pulsos por quarto de nota
    ppq = 24
    beat_duration = 60.0 / bpm
    grid_resolution = beat_duration / ppq
    
    # Quantizar para o grid mais próximo
    quantized_times = np.round(onset_times / grid_resolution) * grid_resolution
    
    return quantized_times

def create_nes_midi(notes: List[dict], output_file: str, style: str = 'NES Original') -> None:
    """Cria arquivo MIDI no estilo selecionado."""
    midi = MidiFile()
    style_params = MUSIC_STYLES[style]
    
    # Criar tracks para cada canal
    tracks = {
        'pulse1': MidiTrack(),  # Melodia principal
        'pulse2': MidiTrack(),  # Harmonia
        'triangle': MidiTrack(), # Baixo
        'noise': MidiTrack()    # Percussão
    }
    
    for track in tracks.values():
        midi.tracks.append(track)
    
    # Normalizar magnitudes
    max_mag = max(note['magnitude'] for note in notes) if notes else 1
    notes.sort(key=lambda x: x['time'])
    
    last_times = {name: 0 for name in tracks}
    
    for note in notes:
        freq = note['frequency']
        midi_note = freq_to_midi_note(freq)
        
        if not (0 <= midi_note <= 127):
            continue
        
        # Selecionar canal e ajustar velocidade baseado no estilo
        if freq < 65:
            channel = 'noise'
            velocity = int(min(127, (note['magnitude'] / max_mag) * style_params['noise_velocity']))
        elif freq < 262:
            channel = 'triangle'
            velocity = int(min(127, (note['magnitude'] / max_mag) * style_params['triangle_velocity']))
        else:
            channel = 'pulse1' if freq % 2 == 0 else 'pulse2'
            velocity = int(min(127, (note['magnitude'] / max_mag) * 
                           style_params[f'{channel}_velocity']))
        
        track = tracks[channel]
        current_time = int(note['time'] * 1000)
        delta = max(0, current_time - last_times[channel])
        last_times[channel] = current_time
        
        # Adicionar nota com velocidade e duração específicas do estilo
        track.append(Message('note_on', note=midi_note, 
                           velocity=max(1, velocity), time=delta))
        
        duration = style_params['note_duration'][channel]
        track.append(Message('note_off', note=midi_note, 
                           velocity=64, time=duration))
    
    midi.save(output_file)

def adjust_audio_tempo(samples: np.ndarray, sr: int, original_bpm: float, target_bpm: float) -> np.ndarray:
    """Ajusta o tempo do áudio para o BPM alvo."""
    tempo_ratio = original_bpm / target_bpm
    return librosa.effects.time_stretch(samples, rate=tempo_ratio)

def process_audio_to_nes(input_file: str, output_file: str, bpm: Optional[float] = None, show_spectral: bool = False) -> None:
    """Processa arquivo de áudio para formato NES."""
    try:
        # 1. Carregar áudio original
        logging.info(f"Carregando arquivo: {input_file}")
        samples, sr = load_audio(input_file)
        
        # 2. Detectar BPM original
        tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
        original_bpm = float(tempo)
        logging.info(f"BPM original detectado: {original_bpm}")
        
        # 3. Ajustar BPM se necessário
        if bpm is not None and abs(bpm - original_bpm) > 1:
            logging.info(f"Ajustando BPM de {original_bpm} para {bpm}")
            samples = adjust_audio_tempo(samples, sr, original_bpm, bpm)
            
            # Salvar áudio temporário com novo BPM
            temp_audio_path = os.path.splitext(input_file)[0] + f"_temp_{int(bpm)}bpm.wav"
            sf.write(temp_audio_path, samples, sr)
            
            # Recarregar o áudio ajustado
            samples, sr = load_audio(temp_audio_path)
            
            # Remover arquivo temporário
            os.remove(temp_audio_path)
            
            logging.info("BPM ajustado com sucesso")
        
        # 4. Análise espectral (opcional)
        if show_spectral:
            analyze_spectral(samples, sr)
        
        # 5. Analisar áudio
        logging.info("Analisando áudio")
        onset_times, pitches, magnitudes = analyze_audio(samples, sr)
        
        # 6. Extrair notas
        logging.info("Extraindo notas")
        notes = extract_notes(pitches, magnitudes, onset_times)
        
        # 7. Criar MIDI
        logging.info("Gerando arquivo MIDI")
        create_nes_midi(notes, output_file)
        
        logging.info("Conversão concluída com sucesso!")
        
    except Exception as e:
        logging.error(f"Erro durante o processamento: {str(e)}")
        raise

def compute_fft(samples, sr):
    window = np.hanning(2048)
    stft = np.abs(librosa.stft(samples, n_fft=2048, hop_length=512, window=window))
    freqs = librosa.fft_frequencies(sr=sr)
    magnitudes = np.mean(stft, axis=1)  # Média das magnitudes ao longo do tempo
    
    # Visualização do espectrograma
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                           y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title("Espectrograma")
    plt.show()
    
    return freqs, stft, magnitudes

def reduce_polyphony(frequencies, magnitudes):
    # Remover frequências com magnitude muito baixa
    threshold = np.mean(magnitudes) * 0.1
    mask = magnitudes > threshold
    
    filtered_frequencies = frequencies[mask]
    filtered_magnitudes = magnitudes[mask]
    
    # Considerar tanto frequência quanto amplitude normalizada
    features = np.column_stack((
        filtered_frequencies / np.max(filtered_frequencies),  # Normalizar frequências
        filtered_magnitudes / np.max(filtered_magnitudes)     # Normalizar magnitudes
    ))
    
    # Realizar clustering apenas nas frequências relevantes
    kmeans = KMeans(n_clusters=4, random_state=0).fit(features)
    return kmeans.labels_, filtered_frequencies

def analyze_audio(samples: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analisa o áudio para extrair características musicais."""
    # Detectar onsets com parâmetros otimizados
    onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        backtrack=True,
        delta=0.2,
        wait=1
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Análise harmônica
    harmonic = librosa.effects.harmonic(samples)
    
    # Extrair pitch e magnitude
    pitches, magnitudes = librosa.piptrack(
        y=harmonic,
        sr=sr,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    
    return onset_times, pitches, magnitudes

def extract_notes(pitches: np.ndarray, magnitudes: np.ndarray, onset_times: np.ndarray) -> List[dict]:
    """Extrai notas significativas do áudio."""
    notes = []
    
    for t in range(pitches.shape[1]):
        if len(onset_times) > 0 and t/pitches.shape[1] * len(onset_times) < len(onset_times):
            onset_idx = int(t/pitches.shape[1] * len(onset_times))
            time = onset_times[onset_idx]
            
            # Encontrar as frequências mais fortes neste frame
            peak_indices = np.argsort(magnitudes[:, t])[-4:][::-1]  # Top 4 frequências
            
            for idx in peak_indices:
                if magnitudes[idx, t] > 0:
                    freq = pitches[idx, t]
                    mag = magnitudes[idx, t]
                    
                    if freq > 0:  # Ignorar frequências zero
                        notes.append({
                            'time': time,
                            'frequency': freq,
                            'magnitude': mag
                        })
    
    return notes

def analyze_spectral(samples: np.ndarray, sr: int) -> None:
    """Analisa e mostra o espectrograma do áudio."""
    # Criar espectrograma
    D = librosa.stft(samples)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plotar espectrograma
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma')
    plt.show()
    
    # Plotar forma de onda
    plt.figure(figsize=(12, 4))
    plt.plot(samples)
    plt.title('Forma de Onda')
    plt.xlabel('Amostras')
    plt.ylabel('Amplitude')
    plt.show()

def main():
    """Interface principal do programa."""
    import tkinter as tk
    from tkinter import filedialog, ttk
    import pygame
    
    def select_file():
        file_path = filedialog.askopenfilename(
            filetypes=[("Arquivos de Áudio", "*.mp3 *.wav")])
        if file_path:
            input_entry.delete(0, tk.END)
            input_entry.insert(0, file_path)
    
    def play_midi():
        if not hasattr(play_midi, "playing"):
            play_midi.playing = False
            
        try:
            if play_midi.playing:
                pygame.mixer.music.stop()
                play_button.config(text="▶ Reproduzir")
                play_midi.playing = False
            else:
                midi_file = os.path.splitext(input_entry.get())[0] + "_nes.mid"
                if os.path.exists(midi_file):
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                    pygame.mixer.music.load(midi_file)
                    pygame.mixer.music.play()
                    play_button.config(text="⏹ Parar")
                    play_midi.playing = True
                    
                    def check_music():
                        if not pygame.mixer.music.get_busy() and play_midi.playing:
                            play_button.config(text="▶ Reproduzir")
                            play_midi.playing = False
                        else:
                            root.after(1000, check_music)
                    
                    check_music()
                else:
                    status_label.config(text="Arquivo MIDI não encontrado!")
        except Exception as e:
            status_label.config(text=f"Erro na reprodução: {str(e)}")
    
    def convert():
        input_file = input_entry.get()
        if not input_file:
            status_label.config(text="Selecione um arquivo de entrada!")
            return
        
        try:
            output_file = os.path.splitext(input_file)[0] + "_nes.mid"
            show_spectral = spectral_var.get()
            selected_style = style_var.get()
            
            status_label.config(text="Convertendo...")
            root.update()
            
            # Carregar e processar áudio
            samples, sr = load_audio(input_file)
            
            if show_spectral:
                analyze_spectral(samples, sr)
            
            onset_times, pitches, magnitudes = analyze_audio(samples, sr)
            notes = extract_notes(pitches, magnitudes, onset_times)
            
            # Criar MIDI com estilo selecionado
            create_nes_midi(notes, output_file, style=selected_style)
            
            status_label.config(text="Conversão concluída!")
            play_button.config(state=tk.NORMAL)
            
        except Exception as e:
            status_label.config(text=f"Erro: {str(e)}")
            logging.error(f"Erro durante o processamento: {str(e)}")
    
    root = tk.Tk()
    root.title("Conversor NES")
    
    # Interface
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    ttk.Label(frame, text="Arquivo de Entrada:").grid(row=0, column=0, sticky=tk.W)
    input_entry = ttk.Entry(frame, width=50)
    input_entry.grid(row=1, column=0, columnspan=2)
    
    ttk.Button(frame, text="Selecionar Arquivo", 
               command=select_file).grid(row=1, column=2)
    
    spectral_var = tk.BooleanVar()
    ttk.Checkbutton(frame, text="Mostrar Análise Espectral", 
                    variable=spectral_var).grid(row=2, column=2)
    
    ttk.Label(frame, text="Estilo Musical:").grid(row=2, column=0, sticky=tk.W)
    style_var = tk.StringVar(value='NES Original')
    style_combo = ttk.Combobox(frame, textvariable=style_var, 
                              values=list(MUSIC_STYLES.keys()),
                              state='readonly', width=15)
    style_combo.grid(row=2, column=1, sticky=tk.W)
    
    ttk.Button(frame, text="Converter", 
               command=convert).grid(row=3, column=0)
    
    play_button = ttk.Button(frame, text="▶ Reproduzir", 
                            command=play_midi, state=tk.DISABLED)
    play_button.grid(row=3, column=1)
    
    status_label = ttk.Label(frame, text="")
    status_label.grid(row=4, column=0, columnspan=3)
    
    root.mainloop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()