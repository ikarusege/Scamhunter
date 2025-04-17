import os
import speech_recognition as sr
from transformers import pipeline
import matplotlib.pyplot as plt
import torch  # PyTorch için

# TensorFlow uyarılarını gizle
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ek TensorFlow uyarılarını gizle

# PyTorch tabanlı pipeline oluştur
try:
    nlp = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt",  # PyTorch kullan
        device=-1  # CPU kullan
    )
    print("✓ NLP modeli başarıyla yüklendi")
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    exit()

# Türkçe riskli kelimeler
risk_keywords = ["para gönder", "hesabın", "hacklendi", "acil", "polis", 
                "ödeme yap", "şifre", "kilitlendi", "torunum", "tutuklandı"]

# Ses tanıma için hazırlık
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Tehdit skoru ve geçmişi
threat_score = 0
score_history = []

def analyze_text(text):
    global threat_score
    print(f"\nAnaliz edilen: {text}")
    
    # Riskli kelimeler
    for keyword in risk_keywords:
        if keyword in text.lower():
            threat_score += 10
            print(f"⚠️ Riskli ifade: '{keyword}' (+10 puan)")
    
    # Duygu analizi
    try:
        result = nlp(text[:512])[0]  # Max 512 karakter
        if result['label'] == 'NEGATIVE' and result['score'] > 0.7:
            threat_score += 5
            print(f"😠 Olumsuz duygu (+5 puan)")
    except Exception as e:
        print(f"AI analiz hatası: {str(e)}")
    
    threat_score = min(threat_score, 100)  # Maksimum %100
    return threat_score

def plot_score():
    plt.clf()
    plt.plot(score_history, 'r-', label='Tehdit Skoru')
    plt.axhline(y=70, color='g', linestyle='--', label='Güvenli Eşik')
    plt.ylim(0, 100)
    plt.title('Gerçek Zamanlı Tehdit Skoru')
    plt.xlabel('Zaman (s)')
    plt.ylabel('Skor (%)')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)

def listen_and_analyze():
    global threat_score
    with microphone as source:
        print("\nDinleniyor... (Çıkmak için Ctrl+C)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        plt.ion()
        
        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = recognizer.recognize_google(audio, language="tr-TR")
                
                threat_score = analyze_text(text)
                print(f"🔴 Güncel Skor: {threat_score}%")
                
                score_history.append(threat_score)
                plot_score()
                
                # Skoru yavaşça düşür
                threat_score = max(0, threat_score - 2)
                
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("Ses anlaşılamadı")
            except KeyboardInterrupt:
                print("\nProgram sonlandırıldı. Son skor:", threat_score)
                break
            except Exception as e:
                print(f"Beklenmeyen hata: {str(e)}")

if __name__ == "__main__":
    try:
        listen_and_analyze()
    finally:
        plt.ioff()
        plt.show()
