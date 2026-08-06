# VLA Literature — Türkçe kaynak kod notları

[Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi)
deposundaki **Vision-Language-Action** modellerinin satır satır Türkçe
incelemesi. Kod değil, okuma defteri.

## Notlar

| Doküman | Model | Satır | Konu |
|---|---|---|---|
| [pi0_fast/pi0_fast_notes.md](pi0_fast/pi0_fast_notes.md) | π0-FAST | 2329 | Otoregresif VLA; DCT tabanlı FAST aksiyon tokenizer'ı |
| [pi0_pytorch/pi0_pytorch_notes.md](pi0_pytorch/pi0_pytorch_notes.md) | π0 (PyTorch) | 1150 | Akış eşleme, action expert, çift-uzman dikkat |
| [pi05/pi05_notes.md](pi05/pi05_notes.md) | π0.5 (JAX) | 466 | `pi05` bayrağı, ayrık state, adaRMSNorm; π0.6 durumu |

## Üç model tek bakışta

| | π0-FAST | π0 | π0.5 |
|---|---|---|---|
| Dosya | `pi0_fast.py` | `pi0.py` | `pi0.py` + `pi05=True` |
| Aksiyon temsili | ayrık `<locNNNN>` | sürekli float | sürekli float |
| Kayıp | cross-entropy | akış eşleme MSE | akış eşleme MSE |
| Action expert | yok | `gemma_300m` | `gemma_300m` |
| `state` kodlaması | metin (256 kutu) | sürekli, `state_proj` | metin (256 kutu) |
| Zaman enjeksiyonu | — | concat + MLP | adaRMSNorm |
| `action_horizon` | 32 | 50 | 50 |
| `max_token_len` | 250 | 48 | 200 |

π0.5, state'i π0-FAST'ten aksiyonu π0'dan alıyor — **iki önceki modelin melezi.**

## Bulgular

- **π0.5 diye ayrı bir dosya yok.** `Pi0Config.pi05: bool` bayrağı ve `pi0.py`
  içinde altı geçiş. Config'in kendi yorumu farkı iki maddede tanımlıyor:
  state ayrık dil token'ı olur, zaman adaRMSNorm ile enjekte edilir.
- **π0.6 açık kaynak değil.** openpi
  [#789](https://github.com/Physical-Intelligence/openpi/issues/789) ·
  [#791](https://github.com/Physical-Intelligence/openpi/issues/791) ·
  [#793](https://github.com/Physical-Intelligence/openpi/issues/793)
  açık ve cevapsız.
- **`stop_gradient` hiç geçmiyor** → Knowledge Insulation bu repodaki eğitim
  kodunda uygulanmamış.
- π0-FAST'in asıl zekâsı model dosyasında değil, **tokenizer'da**.

## Doğrulama

Notlardaki kod alıntıları ham kaynaktan (`raw.githubusercontent.com`)
indirilip `grep -n` ile doğrulandı — özetleyici araç çıktısından değil.
Doğrulandığı commit `upstream.json`'da sabit; haftalık bir Action upstream
sapmasında issue açıyor.

Ayrıntı ve kurallar: [CLAUDE.md](CLAUDE.md)
