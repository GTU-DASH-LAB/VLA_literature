# CLAUDE.md

Bu depo, **Vision-Language-Action (VLA)** modellerinin kaynak kodunu satır satır
inceleyen Türkçe çalışma notlarından oluşuyor. Kod deposu değil — **okuma
defteri**. Buraya çalıştırılabilir kod eklenmez.

## Yazım kuralları

- **Dil: Türkçe.** İngilizce terim ilk geçtiğinde parantezle açıklanır
  (`flow matching` → akış eşleme).
- **Seviye: başlangıç.** Okur transformer'ı biliyor ama robotik VLA'ları
  bilmiyor varsayılır. Atlama yapma, ara adımları yaz.
- **Diyagram kullan.** Mermaid tercih edilir; şekil dönüşümleri için ASCII kutu
  çizimi de olur.
- **Kod alıntıları birebir.** Yeniden yazma, kısaltma, "temizleme" yok. Uzunsa
  ilgili kısmı al ve satır numarasını yaz.

## Doküman iskeleti

Her not dosyası şu sırayı izler:

1. **Kaynak başlığı** — depo, dosya yolu, çekim tarihi, kardeş doküman linkleri.
2. **Doğrulama durumu** — hangi iddialar ham kaynakla teyit edildi (aşağıya bak).
3. Numaralı bölümler — kavramdan koda.
4. **Son bölüm: Soru–Cevap Günlüğü** — her yeni soru buraya `### S<n>` / `**C:**`
   olarak eklenir. Yeni dosya açma, mevcut günlüğe ekle.

## Doğrulama kuralı (bu depo için en önemli kural)

Kaynak kodu **özetleyici araçlarla okuma.** Web çekimi yapan araçlar içeriği
küçük bir modelden geçirip satır düşürebiliyor — satır satır analiz için
işe yaramaz. Ham dosyayı indir:

```bash
curl -sL -o pi0.py https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0.py
```

Sonra `grep -n` ile iddiaları tek tek doğrula ve doküman başındaki
"Doğrulama durumu" bloğuna hangi iddiaların teyit edildiğini yaz.
Doğrulanmamış çıkarımı **kesin dille yazma** — "görünüyor / muhtemelen" kullan.

## Düzeltme kuralı

Eski bir notta hata bulunca **dosyayı sessizce düzeltme.** Doküman başına
`⚠️ **DÜZELTME (tarih):**` bloğu ekle: neyin yanlış olduğu, doğrusu, kaynağı,
ve neyin *değişmediği*. Notlar bir öğrenme kaydı — hata da kaydın parçası.

## Upstream takibi

`upstream.json`, notların doğrulandığı openpi commit'ini ve izlenen dosyaların
blob SHA'larını tutar. Haftalık GitHub Action bunu canlı depoyla karşılaştırır
ve sapma varsa issue açar (`.github/workflows/upstream-drift.yml`).

Elle kontrol:

```bash
gh api repos/Physical-Intelligence/openpi/contents/src/openpi/models/pi0.py --jq .sha
```

Notu güncelledikten sonra `upstream.json`'daki SHA'yı da güncelle — yoksa issue
her hafta yeniden açılır.

## Sabitlenen gerçekler

Tekrar tekrar karıştırılan sayılar. Kaynak: `pi0_config.py`, `pi0_fast.py`.

| | π0 | π0.5 | π0-FAST |
|---|---|---|---|
| `action_dim` | 32 | 32 | 32 |
| `action_horizon` | **50** | **50** | **32** |
| `max_token_len` | 48 | **200** | 250 |
| Kayıp | akış eşleme MSE | akış eşleme MSE | cross-entropy |

- **π0.5 ayrı bir dosya değil** — `Pi0Config.pi05: bool` bayrağı. `pi0.py`
  içinde altı geçişi var (satır 69, 77, 80, 93, 151, 162), hepsi bu.
- **π0.6 açık kaynak değil.** openpi issue #789/#791/#793 açık ve cevapsız.
  `hzm8341/pi0.6` üçüncü taraf, resmî değil.
- `pi0.py`'de **hiç `stop_gradient` yok** → Knowledge Insulation bu repoda
  uygulanmamış (yayınlanan checkpoint'ler için bir şey söylemez).

## Yapma

- Notları özetleyip kısaltma — uzunluk kasıtlı.
- Model ağırlığı, checkpoint, veri seti commit etme. Depo metin kalmalı.
- `.claude/settings.local.json` commit etme (gitignore'da).
