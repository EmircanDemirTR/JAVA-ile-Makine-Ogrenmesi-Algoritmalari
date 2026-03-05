package com.btkakademi.ml.bolum13;

import java.util.Random;

public class QLearningCliffWalking {

    // SABİTLER
    // 4 x 12 grid = 48 durum
    static final int SATIR = 4, SUTUN = 12;
    static final int DURUM_SAYISI = SATIR * SUTUN;

    // Aksiyonlar: yukarı, sağ, aşağı, sol
    static final int AKSIYON_SAYISI = 4;
    static final int YUKARI = 0, SAG = 3, ASAGI = 1, SOL = 2;
    static final String[] AKSIYON_AD = {"YUKARI", "ASAGI", "SOL", "SAG"};
    static final String[] OK = {
            "↑", // YUKARI
            "↓", // ASAGI
            "←", // SOL
            "→"  // SAG
    };

    static final int BASLANGIC = 3 * SUTUN; // (3,0)
    static final int HEDEF = 3 * SUTUN + 11; // (3,11)
    static final int UCURUM_BAS = 3 * SUTUN + 1; // (3,1)
    static final int UCURUM_SON = 3 * SUTUN + 10; // (3,10)

    // Ödüller
    static final double ODUL_UCURUM = -100.0;
    static final double ODUL_ADIM = -1.0;
    static final double ODUL_HEDEF = 0.0;

    // HİPERPARAMETRELER
    static final double ALFA = 0.1; // Öğrenme hızı
    static final double GAMMA = 1.0; // İndirim faktörü
    static final double EPSILON = 0.1; // Keşif oranı

    // EĞİTİM PARAMETRELERİ
    static final int EPISODE_SAYISI = 500;
    static final int MAKS_ADIM = 200;
    static final int RAPOR_ARALIGI = 100;
    static final int SON_N = 50;


    static void main() {
        try {
            double[][] q = new double[DURUM_SAYISI][AKSIYON_SAYISI];
            Random random = new Random();

            double[] oduller = new double[EPISODE_SAYISI];
            int ucurumDusme = 0;

            for (int ep = 00; ep < EPISODE_SAYISI; ep++) {
                int durum = BASLANGIC;
                double toplamOdul = 0.0;
                int adim = 0;

                while (durum != HEDEF && adim < MAKS_ADIM) {
                    int aksiyon = (random.nextDouble() < EPSILON) ?
                            random.nextInt(AKSIYON_SAYISI) : // Keşif
                            enIyiAksiyon(q, durum); // Kullanım

                    int[] sonuc = aksiyonUygula(durum, aksiyon);
                    int yeniDurum = sonuc[0];
                    double odul = sonuc[1];

                    if (odul == ODUL_UCURUM) ucurumDusme++;

                    // Bellman Güncellemesi
                    double maksQ = q[yeniDurum][enIyiAksiyon(q, yeniDurum)];
                    q[durum][aksiyon] += ALFA * (odul + GAMMA * maksQ - q[durum][aksiyon]);

                    durum = yeniDurum;
                    toplamOdul += odul;
                    adim++;
                }

                oduller[ep] = toplamOdul;

                if ((ep + 1) % RAPOR_ARALIGI == 0) {
                    double ort = ortalama(oduller, ep - RAPOR_ARALIGI + 1, ep + 1);
                    System.out.printf("  Episode %3d/%d | Son %d ep ort: %7.1f%n",
                            ep + 1, EPISODE_SAYISI, RAPOR_ARALIGI, ort);
                }

            }
            double sonOrt = ortalama(oduller, EPISODE_SAYISI - SON_N, EPISODE_SAYISI);
            System.out.printf("\n Son %d episode ortalama odul: %.1f%n", SON_N, sonOrt);
            System.out.printf(" Uçurumdan düşme sayısı: %d%n", ucurumDusme);


            // Politika Haritası
            politikaHaritasi(q);

            // Optimal Yol
            System.out.println("Optimal Yol");
            optimalYolYazdir(q);

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

        // Uçurum olursa
        if (yeniDurum >= UCURUM_BAS && yeniDurum <= UCURUM_SON) {
            return new int[]{BASLANGIC, (int) ODUL_UCURUM};
        }

        // Hedef +100
        if (yeniDurum == HEDEF) {
            return new int[]{yeniDurum, (int) ODUL_HEDEF};
        }

        // Normal adım
        return new int[]{yeniDurum, (int) ODUL_ADIM};
    }

    private static double ortalama(double[] dizi, int bas, int son) {
        double toplam = 0.0;
        for (int i = bas; i < son; i++) {
            toplam += dizi[i];
        }
        return toplam / (son - bas);
    }

    private static void politikaHaritasi(double[][] q) {
        for (int i = 0; i < SATIR; i++) {
            System.out.print("  ");
            for (int j = 0; j < SUTUN; j++) {
                int d = i * SUTUN + j;
                if (d == BASLANGIC) System.out.print(" S ");
                else if (d == HEDEF) System.out.print(" G ");
                else if (d >= UCURUM_BAS && d <= UCURUM_SON) System.out.print(" C ");
                else System.out.print(" " + OK[enIyiAksiyon(q, d)] + " ");
            }
            System.out.println();
        }
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
