/*                                           */
/*     Code for Sakai                        */
/*                                           */
/*      ESN for signal classification        */
/*            ( 1-of-k output )              */
/*                                           */
/*       [with bias unit in reservoir]       */
/*  f(sigma*{Y+alpha*(theta*x0+epsilon*s)})  */
/*                                           */
/*    Learning: regularized                  */
/*    Task: NARMA sys.                       */
/*                                           */
/*                         2022 Jan.         */
/*                           by Yoshi        */
/*                                           */

#include <stdio.h>
#include <math.h>
#include <string>
#include <cstdlib>
#include <iostream>



using namespace std;
#define epsilon_conv 1.0e-8

double reservoir(double u);  //リザーバの更新
double f(double y, int typ); //関数f　非線形：tanh(x),線形：100 or -100
double rand_(void);
double grand_(void);    //ガウス　ランダム
double conj_grad(void); //共役勾配法
double narma(int cls);  //関数narma

int f_step; // 最大前方ステップ数 < 100
int idx;
int flag_cut;
int length, tau[10];                           // NARMA param. (tau < length)
double k1[10], k2[10], k3[10], k4[10];         // NARMA param.
double ut0_s[2][10010], yt0_s[2][10010];       // Task data for Training　学習
double ut1_s[2][10010], yt1_s[2][10010];       // Task data for Validation　検証
double ut2_s[2][505][205], yt2_s[2][505][205]; // Task data for Test　テスト

int n_size, type[505];                  // n_size>>ユニットの数？=100で初期化、type>>リザーバのタイプ？ユニットが線形か非線形か
int ic[505][81], k_con;                 //リザーバの結合行列
double x[505], x0[505], y[505];         //ユニットの状態変数、初期値、出力？入力？
double w[505][81], ep[505], theta[505]; //結合重みとか

long flag_over; // Cutoff flag in linear unit　線形単位のカットオフフラグ
double wran, aran, bran, cran;
double pi, pi2;                     // p1=3.14.....,p2=2*p1
double Q[505][505], b[505], a[505]; // Q:matrix_a, b:matrix_b, a[]:方程式の未知数　Aw=b?
double ut[55], yt[55];              //入力値と目標出力値？
double Q0[505][505];                // Q matrix for lambda=0　lamda=0の場合のQ行列

int main()
{
    FILE *fp1, *fp2, *fp3;

    int no, n_rsv, type_rsv;                       // reservoir No., Type　n_rsv>>リザーバの数。=10で初期化。10この平均をとる。 type_rsv=1で初期化>>eservoir type: ESN rand2.(1)
    int t, step[3], wash_out, wash_out_test, mode; // step[0]=1000+wash_out,step[1]=1000+wash_out,step[2]=100+wash_out_test,wash_out=500,wash_out_test=50
    int count_train, count_val, count_test;
    int n, k;
    int unit_idx[505], n_tmp, n_seg;
    int i, j, l;
    int n1, n2;
    int j1, j1_max, j2, j2_max;
    int j1_opt, j2_opt;
    int m, m_max, n_type1[22];
    int m_opt, count;
    int n_reg, lm, lm_opt;
    int j1_test[22], j2_test[22], lm_test[22]; // Test set for each p
    int cls, n_cls, smp, n_smp, c1, c2, c_rsv; // cls>識別する信号の数　n_cls>>信号クラス　n_smp>>(test data sets)/2 =500で初期化　c1>入力信号１　　
    long over[42][42];

    double lambda_val[20], lambda, dpw, pw; // regularization param.正規化パラメータ
    double a_reg[2][20][505];               // weights in regularized learning　
    double w0[505][81], epsilon[505], u;
    double sigma, d_sigma, sigma_min, sigma_max; //状態変数の係数g
    double alpha, d_alpha, alpha_min, alpha_max; //状態変数の係数α
    double u_max, u_min, u_delta;                //入力値？？
    double target_r, t2;                         // target_r>>目標出力値
    double a_opt[2][22][505], t_out, y_out, y_dat[10], t_dat[10];
    double p, dp, p0; // p>>非線形の数
    double err, dty_ave[20], tt_ave, t_ave;
    double nmse[20], nmse_opt, nmse_opt_as, nmse_opt_p;
    double nmse_0[42][42], nmse_reg[42][42]; // [#sig][#alp]; NMSE lambda=0 & opt.
    double seed1, seed2;                     // seed1=1.0>>rand2()、seed2=5.0>>ランダムなESNの実装で初期化

    double mat_err[2][2], y_sum[2], y_max; // error matrix　誤差　行列
    double acc_rate, err_rate;             // accuracy & error rates 正答率、誤差率
    double acc_av[22], err_av[22];



    n_rsv = 10;    // #(Reservoir samples)　リザーバを10個用意して平均とる？
    type_rsv = 1; // Reservoir type: ESN rand2.(1)
    n_size = 100; // # units　ユニットの数
    k_con = 10;   //ユニット一個に結合している結合数？

    n_cls = 2; // #(signal classes)
    wash_out = 1440;
    wash_out_test = 80;
    step[0] = 1440 + wash_out;    // training
    step[1] = 1440 + wash_out;    // validation
    step[2] = 80 + wash_out_test; // test
    n_smp = 18;
    // wash_out=500;
    // wash_out_test=50;
    // step[0]=1000+wash_out;     // training
    // step[1]=1000+wash_out;     // validation
    // step[2]=100+wash_out_test; // test
    // n_smp=500;                 // #(test data sets)/2

    p0 = 0.0;
    dp = 0.1;
    m_max = 10; // # mixture rates; p=#(non.-lin.)/#(total) 線形、非線形の割合？　非線形/総数
    n_reg = 19;
    dpw = 0.25;

    j1_max = 10 - 2; // # sigma values (weight factor)
    d_sigma = 0.1;
    sigma_min = 0.4;
    j2_max = 20; // # alpha values (input signal strength)
    d_alpha = 0.02;
    alpha_min = 0.02;
    sigma_max = sigma_min + d_sigma * (double)j1_max;
    alpha_max = alpha_min + d_alpha * (double)j2_max;

    pi = 3.1415926535897932;
    pi2 = 2.0 * pi;

    seed1 = 1.0; // seed for rand2()
    seed2 = 5.0; // seed for rand2om ESN realization
    wran = seed1;
    aran = 3.2771e4;
    bran = 1.234567891e9;
    cran = 1.0;
    for (i = 1; i <= 31; i++)
        cran = cran * 2.0;

    for (m = 0; m <= m_max; m++)
    {
        acc_av[m] = 0.0;
        err_av[m] = 0.0;
    }

    fp1 = fopen("1ofk_acc.dat", "w");
    fp2 = fopen("1ofk_x0.dat", "w");
    fp3 = fopen("1ofk_x1.dat", "w");

    /*-----------------  Generating time series of narma map  -----------------*/

    n_cls = 2;  // 信号クラスの数
    f_step = 1; // 予測ステップ数

    //----- NARMA-X parameters -----

    length = 30;
    u_max = 0.5; // max of driving term 'u'
    u_min = 0.0; // min of driving term 'u'
    u_delta = u_max - u_min;

    //... パラメータ値（信号１）...
    /*
       tau[0]=9;
       k1[0]=0.3;
       k2[0]=0.05;
       k3[0]=1.5;
       k4[0]=0.01;  // k4=0.1(NARMA10), 0.01(NARMA20)
    */

    tau[0] = 9;
    k1[0] = 0.3;
    k2[0] = 0.05;
    k3[0] = 1.5;
    k4[0] = 0.1; // k4=0.1(NARMA10), 0.01(NARMA20)

    //... パラメータ値（信号２）...

    tau[1] = 8;
    k1[1] = 0.3;
    k2[1] = 0.05;
    k3[1] = 1.5;
    k4[1] = 0.1;

    //... training data ... mode=0(train.)

    mode = 0;
    cls = 0;
    FILE *fp_;
    while (cls <= n_cls - 1)
    {
        t = 0;
        if (cls == 0)
            fp_ = fopen("total-nasi_training.csv", "r");
        // fp_ = fopen("narma.csv", "r");
        if (cls == 1)
            fp_ = fopen("total-musi_n_training.csv", "r");
        // fp_ = fopen("narma2.csv", "r");
        if (fp_ == NULL)
            printf("cannot open file\n");
        flag_cut = 0;
        char input[256], output[256];
        while (fscanf(fp_, "%[^,],%s", input, output) > 1)
        {
            ut0_s[cls][t] = atof(input);
            // if (cls == 1) ut0_s[cls][t] = atof(input) * atof(input);
            // printf("%d, %lf, \n", t, ut0_s[cls][t]);
            t++;
        }
        /*for(i=0; i<=length; i++){ // NARMA-X
          ut[i]=0.0;
          yt[i]=0.0;
        }
        for(t=0; t<=step[mode]+f_step; t++){
          u=u_min+u_delta*rand2();
          for(l=length; l>=1; l--)
            ut[l]=ut[l-1];
          ut[0]=u;
          ut0_s[cls][t]=narma(cls);
        }
   //     for(t=0; t<=step[mode]; t++)
   //       yt0_s[cls][t]=ut0_s[cls][t+f_step];*/
        cls++;
        if (flag_cut != 0)
            cls--;
    }

    //   for(t=0; t<=step[mode]; t++)
    //     fprintf(fp3,"%d %f %f\n",t,ut0_s[0][t],ut0_s[1][t]);

    //... validation data ... mode=1(val.)

    mode = 1;
    cls = 0;
    while (cls <= n_cls - 1)
    {
        t = 0;
        if (cls == 0)
            fp_ = fopen("total-nasi_validation.csv", "r");
        // fp_ = fopen("narma.csv", "r");
        if (cls == 1)
            fp_ = fopen("total-musi_n_validation.csv", "r");
        // fp_ = fopen("narma2.csv", "r");
        if (fp_ == NULL)
            printf("cannot open file\n");
        flag_cut = 0;
        char input[256], output[256];
        while (fscanf(fp_, "%[^,],%s", input, output) > 1)
        {
            ut1_s[cls][t] = atof(input);
            // if (cls == 1) ut1_s[cls][t] = atof(input) * atof(input);
            // printf("%d, %lf, \n", t, ut1_s[cls][t]);
            t++;
        }
        /* for(i=0; i<=length; i++){ // NARMA-X
           ut[i]=0.0;
           yt[i]=0.0;
         }
         for(t=0; t<=step[mode]+f_step; t++){
           u=u_min+u_delta*rand2();
           for(l=length; l>=1; l--)
             ut[l]=ut[l-1];
           ut[0]=u;
           ut1_s[cls][t]=narma(cls);
         }*/
        //     for(t=0; t<=step[mode]; t++)
        //       yt1_s[cls][t]=ut1_s[cls][t+f_step];
        cls++;
        if (flag_cut != 0)
            cls--;
    }

    //   for(t=0; t<=step[mode]; t++)
    //     fprintf(fp3,"%d %f %f\n",t,ut1_s[0][t],ut1_s[1][t]);

    //... test data ... mode=2(test)

    mode = 2;
    for (cls = 0; cls <= n_cls - 1; cls++)
    {
        smp = 1;
        while (smp <= n_smp)
        {
            t = 0;
            if (cls == 0)
                fp_ = fopen("total-nasi_test.csv", "r");
            // fp_ = fopen("narma.csv", "r");
            if (cls == 1)
                fp_ = fopen("total-musi_n_test.csv", "r");
            // fp_ = fopen("narma2.csv", "r");
            if (fp_ == NULL)
                printf("cannot open file\n");
            flag_cut = 0;
            char input[256], output[256];
            int r = 0;
            while (fscanf(fp_, "%[^,],%s", input, output) > 1)
            {
                if (int(r / 160) == smp - 1)
                { // 160行読み込むごとに
                    ut2_s[cls][smp][t] = atof(input);
                    // if (cls == 1) ut2_s[cls][smp][t] = atof(input) * atof(input);
                    // printf("%d, %d %d, %lf\n", cls, smp, t, ut2_s[cls][smp][t]);
                    t++;
                }
                r++;
                if (r >= 160 * n_smp)
                    break;
            }
            /*for(i=0; i<=length; i++){ // NARMA-X
              ut[i]=0.0;
              yt[i]=0.0;
            }
            for(t=0; t<=step[mode]+f_step; t++){
              u=u_min+u_delta*rand2();
              for(l=length; l>=1; l--)
                ut[l]=ut[l-1];
              ut[0]=u;
              ut2_s[cls][smp][t]=narma(cls);
            }*/
            //     for(t=0; t<=step[mode]; t++)
            //       yt2_s[cls][smp][t]=ut2_s[cls][smp][t+f_step];
            smp++;
            if (flag_cut != 0)
                smp--;
        }
    }

    //   smp=n_smp;
    //   for(t=0; t<=step[mode]; t++)
    //     fprintf(fp3,"%d %f %f\n",t,ut2_s[0][smp][t],ut2_s[1][smp][t]);

    /*-----------------  parameters  -----------------*/

    /*----- regularization param. vals. -----*/

    lambda_val[0] = 0.0;
    for (lm = 1; lm <= n_reg; lm++)
    {
        pw = -12.0 + dpw * (double)(lm - 1);
        lambda_val[lm] = pow(10.0, pw); // 10^pw
    }

    /*- Reservoir type -> Data file -*/

    if (type_rsv == 1)
    {
        fprintf(fp1, "# ESN rand2om coupling\n");
        fprintf(fp2, "# ESN rand2om coupling\n");
    }
    else
        printf("Warning: Reservoir type unspecified!\n");

    /*- Parameters -> Data file -*/

    fprintf(fp1, "# n_rsv=%d n_size=%d k_con=%d\n", n_rsv, n_size, k_con);
    fprintf(fp1, "# step: train=%d, val.=%d, test=%d, n_smp=%d\n", step[0], step[1], step[2], n_smp);
    fprintf(fp1, "# alpha: [%f,%f], d_alpha=%f\n", alpha_min, alpha_max, d_alpha);
    fprintf(fp1, "# sigma: [%f,%f], d_sigma=%f\n", sigma_min, sigma_max, d_sigma);
    fprintf(fp1, "# NARMA: f_step=%d\n", f_step);
    // fprintf(fp1,"#  cls=0: tau=%d, k1=%f, k2=%f, k3=%f, k4=%f\n",tau[0],k1[0],k2[0],k3[0],k4[0]);
    // fprintf(fp1,"#  cls=1: tau=%d, k1=%f, k2=%f, k3=%f, k4=%f\n",tau[1],k1[1],k2[1],k3[1],k4[1]);
    fprintf(fp1, "# --- file format ---\n");
    fprintf(fp1, "# p, acc_rate, err_rate, (sigma), (alpha), (lambda)\n");
    fprintf(fp1, "\n");

    /*====================  Loop: Reservoir realizatons (Topology) ====================*/

    for (no = 1; no <= n_rsv; no++)
    { // no-Loop (reservoir ralization) リザーバの作成

        wran = seed2 * (double)no;

        /*----- reservoir parameters -----*/

        //... Bias coeff. ...バイアスの係数

        for (n = 1; n <= n_size; n++)
            theta[n] = 2.0 * (rand_() - 0.5);

        //... Input signal sign ...信号入力

        for (n = 1; n <= n_size; n++)
        {
            if (rand_() < 0.5)
                epsilon[n] = 1.0;
            else
                epsilon[n] = -1.0;
        }

        if (type_rsv == 1)
        {
            //... rand2om-type normalized weights ...　正規化重み
            for (n = 1; n <= n_size; n++)
            {
                n_tmp = n_size;
                for (i = 1; i <= n_size; i++)
                    unit_idx[i] = i;
                for (k = 1; k <= k_con; k++)
                {
                    i = 1 + rand_() * (double)n_tmp;
                    j = unit_idx[i];
                    ic[n][k] = j;                         // coupling: unit n <- unit j　ユニットjからnへの結合
                                                          //          w0[n][k]=grand2()/sqrt((double)k_con);  // Gaussian coupling　　ガウス分布の結合？
                    w0[n][k] = 1.0 / sqrt((double)k_con); // Binary coupling　バイナリ（二進数）結合：コンピュータが処理・記憶するために2進化されたファイルの結合？
                    if (rand_() < 0.5)
                        w0[n][k] = -w0[n][k];

                    if (i != n_tmp)
                        unit_idx[i] = unit_idx[n_tmp];
                    unit_idx[n_tmp] = 0;
                    n_tmp--;
                }
                w0[n][0] = 0.0 * grand_(); // coupling to the bias unit　バイアスユニットへの結合
            }
        }
        else
            printf("Reservoir type undefined!\n");

        /*====================  Loop: p,sigma,alpha  ====================*/

        for (m = 0; m <= m_max; m++)
        { // p-Loop (mixture rate)

            p = p0 + dp * (double)m;
            n_type1[m] = n_size * p; // # nonlin.1 units //非線形ユニットの総数？？
            for (n = 1; n <= n_size; n++)
                type[n] = 0; // type[n]=0 -> lin., 1 -> nonline.1(tanh)
            for (n = 1; n <= n_type1[m]; n++)
                type[n] = 1;

            nmse_opt_as = 1.0e5; // optimal w.r.t. (alpha, sigma)

            // alpha-Loop (input strength)
            for (j2 = 0; j2 <= j2_max; j2++)
            {

                alpha = alpha_min + d_alpha * (double)j2; // input signal strength

                // sigma(g)-Loop (coupling matrix)
                for (j1 = 0; j1 <= j1_max; j1++)
                {

                    sigma = sigma_min + d_sigma * (double)j1; // gain for total input

                    flag_over = 0;

                    for (n = 1; n <= n_size; n++)
                    {
                        w[n][0] = sigma * alpha * theta[n]; // bias unit's coeff.
                        for (k = 1; k <= k_con; k++)        // without bias unit
                            w[n][k] = sigma * w0[n][k];     // (weight)*(stand. dev.)
                    }

                    for (n = 1; n <= n_size; n++)
                        ep[n] = sigma * alpha * epsilon[n]; // (input coeff.)=sigma*alpha*epsilon

                    /*... Covariant mat. & vec. ...*/

                    t2 = 0.0;
                    for (n1 = 0; n1 <= n_size; n1++)
                    {
                        b[n1] = 0.0;
                        for (n2 = 0; n2 <= n_size; n2++)
                            Q[n1][n2] = 0.0;
                    }

                    /*... initial readout weights ...*/

                    for (n = 0; n <= n_size; n++)
                        a[n] = 1.0;

                    /*..... RNN initial condition .....*/

                    x0[0] = 1.0;
                    for (n = 1; n <= n_size; n++)
                        x0[n] = 2.0 * (1.0 - rand_());

                    // x0[0]=1.0;
                    // for(n=1; n<=n_size; n++)
                    // x0[n]=0.0;

                    /*====================  Training phase  ====================*/

                    for (cls = 0; cls <= n_cls - 1; cls++)
                    { // 学習する識別器の番号

                        count_train = 0;

                        for (c1 = 0; c1 <= n_cls - 1; c1++)
                        { // train. data class

                            target_r = 0.0;
                            if (c1 == cls)
                                target_r = 1.0; //信号がc1のとき目標出力値を１で表す

                            for (t = 0; t <= step[0]; t++)
                            {

                                u = ut0_s[c1][t]; // input signal: mode=0 (training)

                                /*..... Reservoir update .....*/
                                if (t % 80 == 0)
                                {
                                    x0[0] = 1.0;
                                    for (n = 1; n <= n_size; n++)
                                        x0[n] = 0.0;
                                }
                                reservoir(u);

                                /*..... Statistical quantities for Error .....*/

                                if (t > wash_out)
                                {
                                    count_train++;

//... Q=<xx> matrix ...
#pragma omp parallel for private(n2)
                                    for (n1 = 0; n1 <= n_size; n1++)
                                    {
                                        for (n2 = n1; n2 <= n_size; n2++)
                                            Q[n1][n2] = Q[n1][n2] + x0[n1] * x0[n2];
                                    }
                                    //... b=<tx> & <tt> ...

                                    //                   target_r=yt0_s[cls][t]; // mode=0 (training)

                                    t2 = t2 + target_r * target_r;
                                    for (n1 = 0; n1 <= n_size; n1++)
                                        b[n1] = b[n1] + target_r * x0[n1]; //∑目標出力y*ユニット状態変数x
                                }

                            } // Loop t (time)

                        } // loop c1 (data class)

                        /*----- Learning & Error comp. for all tasks -----*/

                        for (n1 = 0; n1 <= n_size; n1++)
                        {
                            for (n2 = 0; n2 <= n_size; n2++)
                            {
                                if (n2 < n1)
                                    Q[n1][n2] = Q[n2][n1];
                                Q0[n1][n2] = Q[n1][n2] / (double)count_train; // Q for lambda=0
                            }
                        }

                        t2 = t2 / (double)count_train;
                        for (n1 = 0; n1 <= n_size; n1++)
                            b[n1] = b[n1] / (double)count_train;

                        //... Q matrix ...
                        for (n1 = 0; n1 <= n_size; n1++)
                        {
                            for (n2 = 0; n2 <= n_size; n2++)
                                Q[n1][n2] = Q0[n1][n2];
                        }
                        //... Q+lambda*id ...
                        for (lm = 0; lm <= n_reg; lm++)
                        {
                            lambda = lambda_val[lm];
                            //... Q matrix ...
                            for (n1 = 0; n1 <= n_size; n1++)
                                Q[n1][n1] = Q0[n1][n1] + lambda; // A+λ
                            //... solving normal eq. ...
                            //... reset of readout weights (added) ...
                            for (n = 0; n <= n_size; n++)
                                a[n] = 1.0;
                            //...
                            conj_grad();
                            for (n = 0; n <= n_size; n++)
                                a_reg[cls][lm][n] = a[n]; // weights for (Reg. param.)=lm
                        }

                    } // Loop cls

                    /*====================  Validation phase  ====================*/

                    //... reset of RNN state (added) ...
                    x0[0] = 1.0;
                    for (n = 1; n <= n_size; n++)
                        x0[n] = 2.0 * (1.0 - rand_());
                    // x0[0]=1.0;
                    // for(n=1; n<=n_size; n++)
                    // x0[n]=0.0;
                    //...

                    count_val = 0;
                    tt_ave = 0.0;
                    t_ave = 0.0;
                    for (lm = 0; lm <= n_reg; lm++)
                        dty_ave[lm] = 0.0;

                    for (cls = 0; cls <= n_cls - 1; cls++)
                    { // 誤差測定する識別器の番号
                        for (c1 = 0; c1 <= n_cls - 1; c1++)
                        { // val. data class

                            t_out = 0.0;
                            if (c1 == cls)
                                t_out = 1.0;

                            for (t = 0; t <= step[1]; t++)
                            {

                                u = ut1_s[c1][t]; // mode=1 (validation)

                                //..... Reservoir update .....
                                if (t % 80 == 0)
                                {
                                    x0[0] = 1.0;
                                    for (n = 1; n <= n_size; n++)
                                        x0[n] = 0.0;
                                }
                                reservoir(u);

                                //..... Reservoir output .....

                                if (t > wash_out)
                                {
                                    count_val++;
                                    //                   t_out=yt1_s[cls][t]; // mode=1 (validation);
                                    t_ave = t_ave + t_out;
                                    tt_ave = tt_ave + t_out * t_out;

                                    for (lm = 0; lm <= n_reg; lm++)
                                    {
                                        y_out = 0;
                                        for (n = 0; n <= n_size; n++)
                                            y_out = y_out + a_reg[cls][lm][n] * x0[n];
                                        dty_ave[lm] = dty_ave[lm] + (t_out - y_out) * (t_out - y_out);
                                    }
                                }

                            } // Loop time
                        }     // Loop c1
                    }         // Loop cls

                    nmse_opt = 1.0e5;
                    tt_ave = tt_ave / (double)count_val;
                    t_ave = t_ave / (double)count_val;
                    for (lm = 0; lm <= n_reg; lm++)
                    {
                        dty_ave[lm] = dty_ave[lm] / (double)count_val;
                        nmse[lm] = dty_ave[lm] / (tt_ave - t_ave * t_ave);
                        if (nmse[lm] < nmse_opt)
                        {
                            lm_opt = lm;
                            nmse_opt = nmse[lm];
                        }
                    }

                    //... NMSE data ...

                    over[j1][j2] = flag_over;
                    nmse_0[j1][j2] = nmse[0];        // nmse (lambda=0)
                    nmse_reg[j1][j2] = nmse[lm_opt]; // nmse (lambda=opt.)

                    //... min NMSE over (alpha,sigma) .....

                    err = nmse[lm_opt];
                    if (err < nmse_opt_as)
                    {
                        nmse_opt_as = err; // nmse_opt w.r.t. (alpha,sigma)
                        //... Indices for Test ...
                        j1_test[m] = j1;
                        j2_test[m] = j2;
                        lm_test[m] = lm_opt;
                        for (cls = 0; cls <= n_cls - 1; cls++)
                        {
                            for (n = 0; n <= n_size; n++)
                                a_opt[cls][m][n] = a_reg[cls][lm_opt][n];
                        }
                    }

                } // Loop j1 (sigma)

            } // Loop j2 (alpha)

            j1_opt = j1_test[m];
            j2_opt = j2_test[m];
            alpha = alpha_min + d_alpha * (double)j2_opt;
            sigma = sigma_min + d_sigma * (double)j1_opt;
            lambda = lambda_val[lm_test[m]];

            printf("p=%f NMSE_reg[%d][%d]=%e lambda=%e over=%ld\n",
                   p, j1_opt, j2_opt, nmse_reg[j1_opt][j2_opt], lambda, over[j1_opt][j2_opt]);

            if (m == 0)
            {
                m_opt = m;
                nmse_opt_p = nmse_reg[j1_opt][j2_opt];
            }
            else if (nmse_reg[j1_opt][j2_opt] < nmse_opt_p)
            {
                m_opt = m;
                nmse_opt_p = nmse_reg[j1_opt][j2_opt];
            }

            /*..... Error surface data (fixed p).....*/
            /*
            if(m==5){
              for(j2=0; j2<=j2_max; j2++){
                for(j1=0; j1<=j1_max; j1++){
                  sigma=sigma_min+d_sigma*(double)j1;
                  alpha=alpha_min+d_alpha*(double)j2;
                  fprintf(fp2,"%f %f %e %ld\n",sigma,alpha,nmse_reg[j1][j2],over[j1][j2]);
                }
                fprintf(fp2,"\n");
              }
            }
            */

        } // Loop m (mixture rate)

        /*===============  Test phase: Prediction by optimal (p,alpha,sigma,lambda) ===============*/

        for (m = 0; m <= m_max; m++)
        { // p-Loop (mixture rate)

            //..... reservor setting .....

            p = p0 + dp * (double)m;
            for (n = 1; n <= n_size; n++)
                type[n] = 0; // type[n]=0 -> lin., 1 -> nonline.1(tanh)
            for (n = 1; n <= n_type1[m]; n++)
                type[n] = 1;

            alpha = alpha_min + d_alpha * (double)j2_test[m];
            sigma = sigma_min + d_sigma * (double)j1_test[m];

            for (n = 1; n <= n_size; n++)
            {
                ep[n] = sigma * alpha * epsilon[n];
                w[n][0] = sigma * alpha * theta[n]; // bias unit's coeff.
                for (k = 1; k <= k_con; k++)
                    w[n][k] = sigma * w0[n][k];
            }

            //..... reset of error rate .....

            for (c1 = 0; c1 <= n_cls - 1; c1++)
            {
                for (c2 = 0; c2 <= n_cls - 1; c2++)
                {
                    mat_err[c1][c2] = 0.0;
                }
            }

            count_test = 0;

            for (cls = 0; cls <= n_cls - 1; cls++)
            {

                // bug->    for(c1=0; c1<=n_cls-1; c1++)
                //            p_err[c1]=0.0;

                for (smp = 1; smp <= n_smp; smp++)
                {

                    count_test++;
                    for (c1 = 0; c1 <= n_cls - 1; c1++)
                        y_sum[c1] = 0.0; // Reservoir出力値の和

                    //... reset of RNN state to Zero (added) ...
                    x0[0] = 1.0;
                    for (n = 1; n <= n_size; n++)
                        x0[n] = 0.0;
                    //...

                    for (t = 0; t <= step[2]; t++)
                    {

                        u = ut2_s[cls][smp][t]; // mode=2 (test)

                        //..... Reservoir update .....

                        reservoir(u);

                        //..... Reservoir output .....

                        if (t > wash_out_test)
                        {
                            //                t_out=yt2_s[cls][smp][t]; // mode=2 (test);
                            for (c1 = 0; c1 <= n_cls - 1; c1++)
                            {
                                y_out = 0;
                                for (n = 0; n <= n_size; n++)
                                    y_out = y_out + a_opt[c1][m][n] * x0[n];
                                //                  p_err[c1]=p_err[c1]+(t_out-y_out)*(t_out-y_out);
                                y_sum[c1] = y_sum[c1] + y_out;
                                y_dat[c1] = y_out;

                                t_dat[c1] = 0.0;
                                if (c1 == cls)
                                    t_dat[c1] = 1.0;
                            }
                            if (no == n_rsv && m == m_opt && smp == n_smp && cls == 0)
                            {
                                fprintf(fp2, "%d %e %e %e\n", t, t_dat[0], y_dat[0], y_sum[0]);
                                fprintf(fp3, "%d %e %e %e\n", t, t_dat[1], y_dat[1], y_sum[1]);
                            }
                        }

                    } // Loop time

                    //..... signal class decision .....

                    c_rsv = 0; // RC's class decision
                    y_max = y_sum[0];
                    //            p_min=p_err[0];
                    for (c1 = 1; c1 <= n_cls - 1; c1++)
                    {
                        if (y_sum[c1] > y_max)
                        {
                            c_rsv = c1;
                            y_max = y_sum[c1];
                        }
                    }
                    mat_err[cls][c_rsv]++;

                } // Loop smp
            }     // Loop cls

            for (c1 = 0; c1 <= n_cls - 1; c1++)
            {
                for (c2 = 0; c2 <= n_cls - 1; c2++)
                    mat_err[c1][c2] = mat_err[c1][c2] / (double)count_test;
            }

            acc_rate = 0.0; // accuracy rate 識別率=正解数/データ数
            for (c1 = 0; c1 <= n_cls - 1; c1++)
                acc_rate = acc_rate + mat_err[c1][c1]; //識別率の総数。平均出すために和をとっている？
            err_rate = 1.0 - acc_rate;

            printf("no=%d p=%f Acc_rate=%e\n", no, p, acc_rate);

            alpha = alpha_min + d_alpha * (double)j2_test[m];
            sigma = sigma_min + d_sigma * (double)j1_test[m];
            lambda = lambda_val[lm_test[m]];
            // fprintf(fp1,"%f %e %e %e %e %e\n",p,acc_rate,err_rate,sigma,alpha,lambda);

            acc_av[m] = acc_av[m] + acc_rate;
            err_av[m] = err_av[m] + err_rate;

        } // Loop m (mixture rate)

        printf("m_opt=%d \n", m_opt);

    } // Loop no (Reservoir realization)

    for (m = 0; m <= m_max; m++)
    {
        p = p0 + dp * (double)m;
        acc_av[m] = acc_av[m] / (double)n_rsv; //識別率と誤差率の平均
        err_av[m] = err_av[m] / (double)n_rsv;
        fprintf(fp1, "%f %e %e\n", p, acc_av[m], err_av[m]);
    }

    fclose(fp1);
    fclose(fp2);
    fclose(fp3);

} /*main*/

/*==========   reservoir update   ==========*/

double reservoir(double u) //リザーバの更新
{
    int n, k, j;

#pragma omp parallel for private(k, j) //並列処理
    for (n = 1; n <= n_size; n++)
    {
        y[n] = ep[n] * u;
        for (k = 1; k <= k_con; k++)
        {
            j = ic[n][k];
            y[n] = y[n] + w[n][k] * x0[j];
        }
        y[n] = y[n] + w[n][0] * x0[0]; // including bias unit
    }

    for (n = 1; n <= n_size; n++)
    {
        x[n] = f(y[n], type[n]);
        x0[n] = x[n];
    }
    return 0.0;
}

/*==========   function f   ==========*/

double f(double y, int typ) //関数f
{
    double f;

    if (typ == 1)
        f = tanh(y);
    else
    {
        //..... cutoff linear .....
        if (y < -100.0)
        {
            f = -100.0;
            flag_over++; // printf("over - \n");
        }
        else if (y > 100.0)
        {
            f = 100.0;
            flag_over++; // printf("over + \n");
        }
        else
            f = y;
    }

    return (f);
}

/*==========  subroutine rand2om  ==========*/

double rand_(void)
{
    int m;
    double rnd;

    m = (aran * wran + bran) / cran;
    wran = aran * wran + bran - cran * m;
    rnd = (wran + 0.50) / cran;
    return (rnd);
}

/*==========  subroutine Gauss rand2om  ==========*/

double grand_(void) //ガウス　ランダム
{
    static int iset = 0;
    static double gset;
    double fac, rsq, v1, v2;

    if (iset == 0)
    {
        do
        {
            v1 = 2.0 * rand_() - 1.0;
            v2 = 2.0 * rand_() - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0 || rsq == 0);
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        iset = 1;

        return v2 * fac;
    }
    else
    {
        iset = 0;
        return gset;
    }
}

/*==========  subroutine Conjugate Gradient  ==========*/

double conj_grad(void) //共役勾配法
{
    int i, j, n, dum;
    double alpha, beta, delta;
    double rr, pf;
    double p0[505], r0[505], f0[505], p1[505], r1[505];
    double b_norm, r_norm;
    double aQe, s;

    b_norm = 0.0;
    for (i = 0; i <= n_size; i++)
        b_norm = b_norm + b[i] * b[i];
    b_norm = sqrt(b_norm);

    //... init. vector ...

    for (i = 0; i <= n_size; i++)
    {
        r0[i] = b[i]; // 現ステップの残差ベクトル: r^{k}
        for (j = 0; j <= n_size; j++)
            r0[i] = r0[i] - Q[i][j] * a[j];
        p0[i] = r0[i]; // 現ステップの探索方向ベクトル: p^{k}
    }

    //... conj. direc. optimization ...

    n = 0;
    delta = 1.0;
    r_norm = 1.0e3;
    while (r_norm > epsilon_conv * b_norm && n <= n_size)
    {

        n++;

        //... weight update ...

#pragma omp parallel for private(j) //並列処理
        for (i = 0; i <= n_size; i++)
        {
            f0[i] = 0.0;
            for (j = 0; j <= n_size; j++)
                f0[i] = f0[i] + Q[i][j] * p0[j]; // f=A.p_k
        }

        //... direc. update ...

        rr = 0.0;
        pf = 0.0;
        for (i = 0; i <= n_size; i++)
        {
            rr = rr + r0[i] * r0[i];
            pf = pf + p0[i] * f0[i];
        }
        alpha = rr / pf; // alpha
        delta = 0.0;
        for (i = 0; i <= n_size; i++)
        {
            a[i] = a[i] + alpha * p0[i];   // 近似解の更新: a[i] <-> x^{(k)}
            r1[i] = r0[i] - alpha * f0[i]; // r_{k+1}
            delta = delta + r1[i] * r1[i]; // |r_{k+1}|^2
        }
        beta = delta / rr; // beta
        for (i = 0; i <= n_size; i++)
            p1[i] = r1[i] + beta * p0[i]; // 新しい探索方向ベクトル: p^{k+1}

        for (i = 0; i <= n_size; i++)
        {
            r0[i] = r1[i]; // r1 -> r0
            p0[i] = p1[i]; // p1 -> p0
        }

        delta = sqrt(delta); // |r_{k+1}|

        //... norm of r0

        r_norm = 0.0;
        for (i = 0; i <= n_size; i++)
            r_norm = r_norm + r0[i] * r0[i];
        r_norm = sqrt(r_norm);
    }
    return 0.0;
    // printf("ratio=%e \n",r_norm/b_norm);
}

/*==========   function narma   ==========*/

double narma(int cls) //関数narma
{
    int i;
    double sum;

    sum = 0.0;
    for (i = tau[cls] + 1; i >= 1; i--)
    { // for(i=tau; i>=1; i--){
        yt[i] = yt[i - 1];
        sum = sum + yt[i];
    }

    yt[0] = k1[cls] * yt[1] + k2[cls] * yt[1] * sum + k3[cls] * ut[tau[cls]] * ut[0] + k4[cls];
    if (tau[cls] > 9)
        yt[0] = tanh(yt[0]); // NARMA(tau>=10)
                             /*
                                yt[0]=k1[cls]*yt[1]+k2[cls]*yt[1]*sum+k3[cls]*ut[tau[cls]]*ut[1]+k4[cls];
                                if(tau[cls]>1)
                                  yt[0]=tanh(yt[0]); // NARMA(tau>=10)
                             */
    //... cutt-off bound ...
    if (yt[0] > 1.0)
    {
        yt[0] = 1.0;
        flag_cut = 1;
    }
    else if (yt[0] < -1.0)
    {
        yt[0] = -1.0;
        flag_cut = 1;
    }

    return (yt[0]);
}
