#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <omp.h>

#define PBSTR "===================================================================================================="
#define PBWIDTH 100

void printProgress (double percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}

void getvis(int nf, int nbl, int nsource,  double *f, double *ux, double *uy,
            double *source_flux, double *l, double *m, int nthreads, double complex *vis){
    /*
        Generate visibilities from a list of point sources and their apparent flux densities.

        Parameters
        ----------
        f : double[nf]
            Frequencies of observation, normalised by reference frequency.

        ux, uy : double[nbl]
            The x,y co-ordinates of the baselines, in units of wavelengths, at reference frequency (f=1).

        source_flux : double[nsource]
            The *apparent* flux of each point-source.

        l,m : double[nsource]
            Positions, in sin-projected units, of the sources on the sky.

        Returns
        -------
        vis : complex[nbl * nf]
            The complex visibilities (filled within the function).
    */
    int i,j,k;
    double arg;

    complex double thisval, sum;

    omp_set_num_threads(nthreads);
    int done = 0;

    #pragma omp parallel for collapse(2) private(arg,j,sum)
    for(i=0;i<nf;i++){
        for(k=0;k<nbl;k++){
            sum = 0.0;
            for(j=0;j<nsource;j++){
                arg = -f[i]*(ux[k]*l[j] + uy[k]*m[j]);
                sum += source_flux[j] * cexp(I*arg);
            }
            vis[i*nbl + k] = sum;

        }
        done += 1;
        printProgress(done/nf);
    }
}


long int get_baselines(unsigned int n_ant, double *x, double *y, double diameter, double *bl_x, double *bl_y){
    long int nbl = 0;

    int i, j;
    double diameter2 = diameter*diameter;
    double dx, dy;

    for(i=0;i<n_ant;i++){
        for(j=i+1;j<n_ant;j++){

            dx = x[i] - x[j];
            dy = y[i] - y[j];

            if ((dx*dx + dy*dy) >= diameter2){
                bl_x[nbl] = dx;
                bl_y[nbl] = dy;
                nbl++;
            }else{
                printf("\n\n\n");
                return -nbl;
            }
        }
    }
    return nbl;
}


void get_bad_antennas(unsigned int n_ant, double *x, double *y, double diameter, int start, int *bad_antennas){

    long int nbl = 0;

    int i, j;
    double diameter2 = diameter*diameter;
    double dx, dy;

    for(i=0;i<n_ant;i++){
        for(j=i+1;j<n_ant;j++){
            nbl++;

            if (nbl>start && bad_antennas[j] == 0 && bad_antennas[i] == 0){
                dx = x[i] - x[j];
                dy = y[i] - y[j];

                if ((dx*dx + dy*dy) <= diameter2){
                    bad_antennas[j] = 1;
                }

            }
        }
    }
}

