import pandas as pd
# Kütüphaneyi içe aktarın ve veri kümesini yükleyin
from datasets import load_dataset
from helpers.helpers import *

pd.pandas.set_option('display.max_columns', None)

# Sentiment140 veri kümesini yükleyin
dataset = load_dataset("sentiment140")

# Eğitim verisini gözlemleyin
# print(dataset['train'][0:100])

# Veri kümesindeki bazı temel istatistikleri görüntüleyin
# print("Toplam örnek sayısı:", dataset['train'].num_rows)
# print("Özellikler:", dataset['train'].column_names)


# DataFrame'e dönüştürmek için önce listelere çevirin
data = {key: dataset['train'][key] for key in dataset['train'].column_names}

# Listeleri kullanarak DataFrame oluşturun
df = pd.DataFrame(data)
df = df.loc[:10000]

# DataFrame'i kontrol edin
# df.head(100)
# check_df(df)

import re


# EDA
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)  # URL'leri kaldır
    text = re.sub(r'@\w+', '', text)  # Kullanıcı adlarını kaldır
    text = re.sub(r'#\S+', '', text)  # Hashtag'leri kaldır
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    text = text.lower()  # Metni küçük harfe çevir
    return text


# DataFrame üzerinde temizleme işlemini uygulayın
df['clean_text'] = df['text'].apply(clean_text)

# Dil Tespiti
from langdetect import detect


def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'


detect_language(df['clean_text'])

df['language'] = df['clean_text'].apply(detect_language)
# df['language'].value_counts() # hangi dilden kaç tane yorum olduğu

df = df[df['language'] == 'en']  # Sadece İngilizce metinleri filtrele

# Tokenizasyon ve Durdurma Kelimeleri
import nltk  # NLTK (Natural Language Toolkit)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK kütüphanesi için gerekli dil işleme modellerinin ve veri setlerinin indirilmesi
nltk.download('punkt')  # Cümle ve kelime tokenleştirme modelleri için
nltk.download('stopwords')  # Çeşitli dillerde durdurma kelimelerinin listesi için

# İngilizce dilindeki durdurma kelimelerinin bir kümesinin oluşturulması
stop_words = set(stopwords.words('english'))

# Olumsuzluk ekleri içeren kelimeleri listeden çıkar - bu ekler önemli olabilir
negations = {"no", "not", "nor", "none", "never", "n't", "cannot", "shouldn't", "isn't", "wasn't", "aren't", "don't",
             "didn't"}
filtered_stop_words = stop_words.difference(negations)


# Metni kelimelere ayıran ve durdurma kelimelerini çıkaran fonksiyon
def tokenize(text):
    # Metni kelimelere ayırma
    words = word_tokenize(text)
    # Durdurma kelimeleri olmayan kelimeleri filtreleme ve listeleme
    filtered_words = [word for word in words if word not in filtered_stop_words]
    # Filtrelenmiş kelimeleri döndürme
    return filtered_words


# DataFrame üzerinde 'clean_text' sütununu kullanarak tokenize fonksiyonunun uygulanması
# ve sonuçların 'tokens' adlı yeni bir sütunda saklanması
df['tokens'] = df['clean_text'].apply(tokenize)

df.head()
###################################################################
# PyTorch Dataset Sınıfı Oluşturma

from torch.utils.data import Dataset
import torch


class TwitterSentimentDataset(Dataset):
    """Sentiment140 veri kümesi için özel Dataset sınıfı"""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return {"text": text, "label": torch.tensor(label, dtype=torch.long)}


# DataLoader
from torch.utils.data import DataLoader

"""
DataLoader, veri setinizi belirtilen batch_size değeri kadar parçalara ayırarak modelinize yığınlar halinde sunar. 
Veri yükleme işlemleri sırasında CPU ve GPU kaynaklarının etkin kullanımını destekler. Özellikle, çoklu işlem (multi-processing) desteği sayesinde veri yükleme işlemleri daha hızlı gerçekleştirilebilir.

shuffle=True parametresi, her eğitim epoch'u başlamadan önce veri setinin karıştırılmasını sağlar.
Bu, modelin eğitim sürecinde verilere olan bağımlılığını ve ezberlemesini önlemeye yardımcı olur, böylece modelin genelleştirme yeteneğini artırır. 

PyTorch'un DataLoader sınıfı, num_workers adlı bir parametre alır. Bu parametre, veri yükleme işlemini gerçekleştirecek iş parçacığı (worker) sayısını belirtir
# num_workers=4 veri yükleme işlemi dört iş parçacığı kullanacak
"""

# DataLoader örneği oluşturma
batch_size = 32
shuffle = True  # Verileri karıştır
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

###################################################################
# Model Seçimi
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn

# Önceden eğitilmiş bir tokenizer yükleyin
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tokenize işlevi, metni modelin anlayabileceği token ID'lerine çevirir
# tokenize fonksiyonu, metinleri tokenlere çevirme ve bu tokenleri bir sinir ağı modelinin işleyebileceği forma getirme işlemini yerine getirir.

def tokenize(text):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']


# PyTorch Dataset Sınıfı
# Twitter verilerini bir Dataset sınıfında yönetme
class TwitterDataset(Dataset):
    # Sınıfın başlatıcı metodu, veri seti özelliklerini başlatır
    def __init__(self, tweets, labels, tokenizer, max_len):
        self.tweets = tweets  # Tweet listesi
        self.labels = labels  # Her tweet için etiket listesi
        self.tokenizer = tokenizer  # Tokenizer nesnesi (BERT için)
        self.max_len = max_len  # Tokenizasyon için maksimum uzunluk

    # Veri setinin uzunluğunu döndürür (toplam tweet sayısı)
    def __len__(self):
        return len(self.tweets)

    # Veri setinden belirli bir index'e karşılık gelen öğeyi döndürür
    def __getitem__(self, item):
        tweet = str(self.tweets[item])  # Belirli bir tweet
        label = self.labels[item]  # Belirli bir tweet için etiket

        # Tweet'i tokenize etme ve encoding işlemi
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,  # Özel tokenlar ekler (örn. [CLS], [SEP])
            max_length=self.max_len,  # Token maksimum uzunluğunu belirler
            return_token_type_ids=False,  # Token tip ID'lerini döndürmez
            pad_to_max_length=True,  # Maksimum uzunluğa kadar padding uygular
            return_attention_mask=True,  # Dikkat maskesini hesaplar
            return_tensors='pt',  # PyTorch tensorleri olarak döndürür
        )

        return {
            'tweet_text': tweet,  # Orijinal tweet metni
            'input_ids': encoding['input_ids'].flatten(),  # Input ID'ler
            'attention_mask': encoding['attention_mask'].flatten(),  # Dikkat maskesi
            'labels': torch.tensor(label, dtype=torch.long)  # Etiketler
        }


# Veri Seti Örneği Oluşturma

# Metinleri ve etiketleri çıkar
texts = df['text'].tolist()
labels = df['sentiment'].tolist()  # 'sentiment' sütunu duygu etiketlerini içeriyor varsayalım

# Metinleri tokenize et
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

# Dataset örneği oluştur
sentiment_dataset = TwitterSentimentDataset(encodings, labels)


# Model Eğitimi Fonksiyonu
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()  # Modeli eğitim moduna alır

    losses = []  # Her batch için kayıpları saklar
    correct_predictions = 0  # Doğru tahmin sayısını tutar

    for d in data_loader:
        input_ids = d["input_ids"].to(device)  # Input ID'leri cihaza gönderir
        attention_mask = d["attention_mask"].to(device)  # Dikkat maskesini cihaza gönderir
        labels = d["labels"].to(device)  # Etiketleri cihaza gönderir

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss  # Hesaplanan kaybı alır
        _, preds = torch.max(outputs.logits, dim=1)  # Logitlerden tahminleri alır
        correct_predictions += torch.sum(preds == labels)  # Doğru tahminleri sayar
        losses.append(loss.item())  # Kayıp listesine kaybı ekler

        loss.backward()  # Gradyan hesaplaması yapar
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradyanları kırpar
        optimizer.step()  # Optimizasyon adımını uygular
        scheduler.step()  # Learning rate scheduler adımını uygular
        optimizer.zero_grad()  # Gradyanları sıfırlar

    return correct_predictions.double() / n_examples, np.mean(losses)  # Doğruluk ve ortalama kaybı döndürür


# Dataset örneği oluşturma
dataset = TwitterDataset(tweets=tweets, labels=labels, tokenizer=tokenizer, max_len=128)

##########################################################


# Model, Optimizer ve Scheduler Kurulumu
# Model loss function (kayıp fonksiyonu), optimizer ve eğitim için kullanılacak cihazı (CPU veya GPU) ayarlanması.

from transformers import BertForSequenceClassification, AdamW
from torch.optim.lr_scheduler import StepLR
import torch

# Cihazı ayarlama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT modelini yükleme ve ayarlama
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Örneğin, pozitif ve negatif için 2 sınıf varsayıyoruz
)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=1000,
                   gamma=0.1)  # Adım sayısı ve gamma değeri örnek olarak verilmiştir, duruma göre ayarlanmalıdır.

# Kayıp fonksiyonu
loss_fn = torch.nn.CrossEntropyLoss()

##########################################################
# Eğitim Döngüsü


# Eğitim için epoch sayısı
num_epochs = 3

for epoch in range(num_epochs):
    # Train fonksiyonu çağrısı
    correct_predictions, average_loss = train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(dataset)
    )

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.5f}, Accuracy: {correct_predictions:.2f}')

    # Scheduler güncellemesi (eğer varsa)
    scheduler.step()

# Model Değerlendirme

# Test veri seti için DataLoader oluşturma ve kullanma (örnek veri seti ve DataLoader varsayılmıştır)
test_data_loader = DataLoader(dataset, batch_size=32, shuffle=False)


def evaluate_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

    return correct_predictions.double() / total_predictions


accuracy = evaluate_model(model, test_data_loader, device)
print(f'Test Accuracy: {accuracy:.2f}')
