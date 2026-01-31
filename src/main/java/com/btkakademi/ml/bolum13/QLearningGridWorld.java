package com.btkakademi.ml.bolum13;

import java.util.Random;

public class QLearningGridWorld {

    // SABİTLER
    // Grid boyutları 5x5
    static final int SATIR = 5, SUTUN = 5;
    static final int DURUM_SAYISI = SATIR * SUTUN;

    // AKSİYONLAR - 4 YÖN
    static final int AKSIYON_SAYISI = 4;
    static final int YUKARI = 0;
    static final int ASAGI = 1;
    static final int SOL = 2;
    static final int SAG = 3;
    static final String[] AKSIYON_AD = {"YUKARI", "ASAGI", "SOL", "SAG"};

    // ÖZEL DURUMLAR
    static final int BASLANGIC = 0; // (0,0)
    static final int HEDEF = 24;   // (4,4)
    static final int[] ENGELLER = {12, 18}; // (2,2) ve (3,3)

    // ÖDÜLLER
    static final double ODUL_HEDEF = 100.0;
    static final double ODUL_ADIM = -1.0;

    // Q-ÖĞRENME PARAMETRELERİ
    static final double ALFA = 0.1; // α
    static final double GAMA = 0.9; // γ
    static final double EPSILON = 0.1; // ε (Keşif oran

    // EĞİTİM PARAMETRELERİ
    static final int EPISODE_SAYISI = 1000;
    static final int MAKS_ADIM = 100;
    static final int RAPOR_ARALIGI = 100;


    static void main() {
        try {
            double[][] qTablosu = new double[DURUM_SAYISI][AKSIYON_SAYISI];

            Random random = new Random(42);
            double[] episodeOdulleri = new double[EPISODE_SAYISI];
            int[] episodeAdimlari = new int[EPISODE_SAYISI];
            boolean[] basarili = new boolean[EPISODE_SAYISI];

            for (int ep = 0; ep < EPISODE_SAYISI; ep++) {
                int durum = BASLANGIC;
                double toplamOdul = 0.0;
                int adim = 0;

                while (durum != HEDEF && adim < MAKS_ADIM) {
                    int aksiyon = (random.nextDouble() < EPSILON) ?
                            random.nextInt(AKSIYON_SAYISI) :
                            enIyiAksiyon(qTablosu, durum);

                    int[] sonuc = aksiyonUygula(durum, aksiyon);
                    int yeniDurum = sonuc[0];
                    double odul = sonuc[1];

                    //Bellman Güncellemesi
                    double maksQ = qTablosu[yeniDurum][enIyiAksiyon(qTablosu, yeniDurum)];
                    double tdHata = (odul + GAMA * maksQ) - qTablosu[durum][aksiyon];
                    qTablosu[durum][aksiyon] += ALFA * tdHata;

                    durum = yeniDurum;
                    toplamOdul += odul;
                    adim++;
                }

                episodeOdulleri[ep] = toplamOdul;
                episodeAdimlari[ep] = adim;
                basarili[ep] = (durum == HEDEF);

                if ((ep + 1) % RAPOR_ARALIGI == 0) {
                    raporYazdir(ep, episodeOdulleri, episodeAdimlari, basarili);
                }
            }

            System.out.println("YAKINLAMA");
            double ilkOrt = ortalama(episodeOdulleri, 0, RAPOR_ARALIGI);
            double sonOrt = ortalama(episodeOdulleri, EPISODE_SAYISI - RAPOR_ARALIGI, EPISODE_SAYISI);

            System.out.println("İlk 100 ep ort. odul " + ilkOrt);
            System.out.println("Son 100 ep ort. odul " + sonOrt);
            System.out.println(sonOrt > ilkOrt ?
                    "Öğrenme başarılı!" :
                    "Öğrenme başarısız!");


            // Optimal Yol
            System.out.println("OPTİMAL YOL");

            optimalYolYazdir(qTablosu);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static int enIyiAksiyon(double[][] q, int durum) {
        int en = 0;
        for (int a = 1; a < AKSIYON_SAYISI; a++) {
            if (q[durum][a] > q[durum][en]) {
                en = a;
            }
        }
        return en;
    }

    private static int[] aksiyonUygula(int durum, int aksiyon) {
        int satir = durum / SUTUN;
        int sutun = durum % SUTUN;

        int ySatir = satir, ySutun = sutun;

        switch (aksiyon) {
            case YUKARI -> ySatir--;
            case ASAGI -> ySatir++;
            case SOL -> ySutun--;
            case SAG -> ySutun++;
        }

        if (ySatir < 0 || ySatir >= SATIR || ySutun < 0 || ySutun >= SUTUN) {
            return new int[]{durum, (int) ODUL_ADIM};
        }

        int yeniDurum = ySatir * SUTUN + ySutun;

        // Engel olursa
        if (yeniDurum == ENGELLER[0] || yeniDurum == ENGELLER[1]) {
            return new int[]{durum, (int) ODUL_ADIM};
        }

        // Hedef +100
        if (yeniDurum == HEDEF) {
            return new int[]{yeniDurum, (int) ODUL_HEDEF};
        }

        // Normal adım
        return new int[]{yeniDurum, (int) ODUL_ADIM};
    }

    private static void raporYazdir(int ep, double[] oduller, int[] adimlar, boolean[] basarili) {
        int basari = 0;
        double ortOdul = 0, ortAdim = 0;
        for (int i = ep - RAPOR_ARALIGI + 1; i <= ep; i++) {
            ortOdul += oduller[i];
            ortAdim += adimlar[i];
            if (basarili[i]) {
                basari++;
            }
            System.out.println(String.format(" Episode %4d/%d | Ort. Odul=%7.1f, Ort. Adim=%5.1f | Basarili=%3d/%d",
                    ep + 1, EPISODE_SAYISI, ortOdul / RAPOR_ARALIGI, ortAdim / RAPOR_ARALIGI, basari, RAPOR_ARALIGI));
        }
    }

    private static double ortalama(double[] dizi, int bas, int son) {
        double toplam = 0.0;
        for (int i = bas; i < son; i++) {
            toplam += dizi[i];
        }
        return toplam / (son - bas);
    }

    private static void optimalYolYazdir(double[][] q) {

        int durum = BASLANGIC, adim = 0;
        boolean[] ziyaret = new boolean[DURUM_SAYISI];

        while (durum != HEDEF && adim < MAKS_ADIM) {
            if (ziyaret[durum]) {
                System.out.println("Döngü tespit edildi, çıkılıyor.");
                break;
            }
            ziyaret[durum] = true;

            int aksiyon = enIyiAksiyon(q, durum);
            System.out.println("Adım " + adim + ": Durum " + durum + " -> Aksiyon " + AKSIYON_AD[aksiyon]);

            durum = aksiyonUygula(durum, aksiyon)[0];
            adim++;
        }

        if (durum == HEDEF) {
            System.out.println("Hedefe ulaşıldı!");
        }

    }
}
