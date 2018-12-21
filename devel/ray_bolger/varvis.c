#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <omp.h>

#define PI 3.141592653589

int var_vis(int nf, int nomega, int nui, double *f, double *omega, double *ui, double *taper, double sigma, double u,
            double extent, double *res, int nthreads){

    double q2 =  PI*PI * sigma*sigma;

    // get weights
    double *w_nu_ui = calloc(nf*nui, sizeof(double));
    double *W_nu = calloc(nf, sizeof(double));

    double thisvar, wnuui, wnuui_tot, pdist;
    int i,j, iom, k, m, started, started_i, started_m;
    double fdepj, fdep;

    double complex this_one, two_pi_om, kernel;

    extent = extent/(sqrt(2)*PI*sigma);

    // Fill up weights
    for(i=0;i<nf;i++){
        started=0;
        for(j=0;j<nui;j++){
            thisvar = u - f[i]*ui[j];
            if (fabsf(thisvar) < extent){
                w_nu_ui[i*nui + j] = exp(-2*q2*thisvar*thisvar);
                W_nu[i] += w_nu_ui[i*nui + j];
                started=1;
            }else if(started){
    //            printf("For freq %d stopping at %d\n", i, j);

                break;
            }
        }
    }



    omp_set_num_threads(nthreads);

    #pragma omp parallel for private(two_pi_om, this_one, j, started_i, fdepj, i, wnuui, k, started_m, fdep, kernel, m, wnuui_tot, pdist)
    for(iom=0;iom<nomega;iom++){
        this_one = 0;
        printf("Doing %d of %d iterations, %d\n", iom, nomega, omp_get_thread_num());

        two_pi_om = -2*I*PI*omega[iom];
        for(j=0;j<nf;j++){
            started_i = 0;
            fdepj = taper[j] / W_nu[j];

            for(i=0;i<nui;i++){
                wnuui =  w_nu_ui[j*nui + i];
                if (wnuui > 0){

                    for(k=0;k<nf;k++){
                        started_m = 0;
                        fdep = fdepj * taper[k] / W_nu[k];
                        kernel = cexp(two_pi_om * (f[j]-f[k]));

                        for(m=0;m<nui;m++){
                            wnuui_tot = wnuui * w_nu_ui[k*nui + m];
                            if(wnuui_tot > 0){
                                pdist = f[j]*ui[i] - f[k]*ui[m];
                                this_one += fdep * kernel * wnuui_tot * exp(-q2 * pdist*pdist);
        //                        printf("jf=%d ibl=%d kf=%d mbl=%d (%lf, %lf) || %lf (%lf %lf) %lf %lf\n", j, i, k, m, creal(this_one),
        //                               cimag(this_one), fdep, creal(kernel), cimag(kernel), wnuui, exp(-q2 * pdist*pdist));
        //                        printf("   %lf %lf %lf %lf\n", taper[j], taper[k], W_nu[j], W_nu[k]);
                                started_m = 1;
                            }else if (started_m){
                                if (k<3 && j<3)
          //                      printf("For m: freq %d stopping at %d (%d)\n", k, m, started_m);
                                break;
                            }
                        }
                    }
                    started_i = 1;
                } else if (started_i){
                    if(j<3)
            //        printf("For i: freq %d stopping at %d\n", j, i);
                    break;
                }
            }
        }
        res[iom] += creal(this_one);
    }

    return(1);
}