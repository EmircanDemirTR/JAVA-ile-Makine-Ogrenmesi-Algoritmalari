package com.btkakademi.ml.bolum13;

import java.util.Random;

public class QLearningHiperparametre {
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
            double[] baseOd = new double[EPISODE_SAYISI];
            egit(0.1, 1.0, 0.1, false, 0.1, baseOd, new Random(42));
            System.out.println("Son " + SON_N + " episode ort. odul: " + String.format("%.1f", sonNOrt(baseOd)) + "\n");

            // Alpha parametresi ile denemeler
            for (double alpha : new double[]{0.01, 0.1, 0.5, 0.9}) {
                double[] od = new double[EPISODE_SAYISI];
                egit(alpha, 1.0, 0.1, false, 0.1, od, new Random(42));
                System.out.println(String.format(" Alpha=%.2f ile son %d episode ort. odul: %.1f",
                        alpha, SON_N, sonNOrt(od)));
            }


            // Epsilon parametresi ile denemeler
            for (double eps : new double[]{0.01, 0.1, 0.5, 0.9}) {
                double[] od = new double[EPISODE_SAYISI];
                egit(0.1, 1.0, eps, false, eps, od, new Random(42));
                System.out.println(String.format(" Epsilon=%.2f ile son %d episode ort. odul: %.1f",
                        eps, SON_N, sonNOrt(od)));
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static double[][] egit(double alpha, double gamma,
                                   double epsBas, boolean epsAzalt, double epsBit,
                                   double[] episodeOudelleri, Random random) {
        double[][] q = new double[DURUM_SAYISI][AKSIYON_SAYISI];

        for (int ep = 0; ep < EPISODE_SAYISI; ep++) {
            int durum = BASLANGIC;
            double toplamOdul = 0;

            double epsilon = epsAzalt ?
                    epsBas + ((double) ep / (EPISODE_SAYISI - 1)) * (epsBit - epsBas)
                    : epsBas;

            int adim = 0;
            while ((durum != HEDEF) && (adim < MAKS_ADIM)) {
                int aksiyon = (random.nextDouble() < epsilon) ?
                        random.nextInt(AKSIYON_SAYISI)
                        : enIyiAksiyon(q, durum);

                int[] sonuc = aksiyonUygula(durum, aksiyon);
                int yeniDurum = sonuc[0];
                double odul = sonuc[1];

                //Bellman denklemi
                double maksQ = q[yeniDurum][enIyiAksiyon(q, yeniDurum)];
                q[durum][aksiyon] += alpha * (odul + gamma * maksQ - q[durum][aksiyon]);

                durum = yeniDurum;
                toplamOdul += odul;
                adim++;
            }
            episodeOudelleri[ep] = toplamOdul;
        }
        return q;
    }

    private static int[] aksiyonUygula(int durum, int aksiyon) {
        int satir = durum / SUTUN, sutun = durum % SUTUN;
        int ySatir = satir, ySutun = sutun;

        switch (aksiyon) {
            case YUKARI -> ySatir--;
            case ASAGI -> ySatir++;
            case SOL -> ySutun--;
            case SAG -> ySutun++;
        }

        if (ySatir < 0 || ySatir >= SATIR || ySutun < 0 || ySutun >= SUTUN)
            return new int[]{durum, (int) ODUL_ADIM};

        int yeni = ySatir * SUTUN + ySutun;
        if (yeni >= UCURUM_BAS && yeni <= UCURUM_SON) return new int[]{BASLANGIC, (int) ODUL_UCURUM};
        if (yeni == HEDEF) return new int[]{yeni, (int) ODUL_HEDEF};
        return new int[]{yeni, (int) ODUL_ADIM};
    }

    private static int enIyiAksiyon(double[][] q, int durum) {
        int en = 0;
        for (int a = 1; a < AKSIYON_SAYISI; a++)
            if (q[durum][a] > q[durum][en]) en = a;
        return en;
    }

    private static double sonNOrt(double[] od) {
        double t = 0;
        for (int i = EPISODE_SAYISI - SON_N; i < EPISODE_SAYISI; i++) t += od[i];
        return t / SON_N;
    }

    private static void optimalYolYazdir(double[][] q) {
        int durum = BASLANGIC, adim = 0;
        boolean[] ziyaret = new boolean[DURUM_SAYISI];

        while (durum != HEDEF && adim < MAKS_ADIM) {
            if (ziyaret[durum]) {
                System.out.println("  [!] Dongu tespit edildi!");
                break;
            }
            ziyaret[durum] = true;

            int aksiyon = enIyiAksiyon(q, durum);
            System.out.println(String.format("  Adim %2d: (%d,%2d) → %s [Q=%.2f]",
                    adim + 1, durum / SUTUN, durum % SUTUN,
                    AKSIYON_AD[aksiyon], q[durum][aksiyon]));

            durum = aksiyonUygula(durum, aksiyon)[0];
            adim++;
        }

        if (durum == HEDEF) {
            System.out.println(String.format("  Adim %2d: (%d,%2d) → HEDEF!",
                    adim + 1, durum / SUTUN, durum % SUTUN));
            System.out.println("Toplam: " + adim + " adim (optimal ~13)");
        }
    }

}
