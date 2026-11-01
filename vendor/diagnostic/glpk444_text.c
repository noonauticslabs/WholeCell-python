/* SPDX-License-Identifier: GPL-3.0-or-later
 * Standalone text-protocol driver for GLPK 4.44.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <glpk.h>

static double read_double(FILE *f) {
    double x;
    if (fscanf(f, "%lf", &x) != 1) { fprintf(stderr, "bad fixture data\n"); exit(4); }
    return x;
}

static void print_value(double x) {
    if (isfinite(x)) printf("%.17g", x); else printf("null");
}

static void print_vector(glp_prob *p, int n, int is_col, int which) {
    putchar('[');
    for (int k = 1; k <= n; ++k) {
        if (k > 1) putchar(',');
        double x;
        if (which == 0) x = is_col ? glp_get_col_prim(p, k) : glp_get_row_prim(p, k);
        else x = is_col ? glp_get_col_dual(p, k) : glp_get_row_dual(p, k);
        print_value(x);
    }
    putchar(']');
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s fixture.txt\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 3; }
    int m, n, maximize;
    if (fscanf(f, "%d %d %d", &m, &n, &maximize) != 3 || m <= 0 || n <= 0) return 4;
    double *obj = calloc((size_t)n, sizeof(double));
    double *lb = calloc((size_t)n, sizeof(double));
    double *ub = calloc((size_t)n, sizeof(double));
    double *rhs = calloc((size_t)m, sizeof(double));
    char *ct = calloc((size_t)m, sizeof(char));
    char *vt = calloc((size_t)n, sizeof(char));
    for (int j = 0; j < n; ++j) obj[j] = read_double(f);
    for (int j = 0; j < n; ++j) lb[j] = read_double(f);
    for (int j = 0; j < n; ++j) ub[j] = read_double(f);
    for (int i = 0; i < m; ++i) rhs[i] = read_double(f);
    for (int i = 0; i < m; ++i) if (fscanf(f, " %c", &ct[i]) != 1) return 4;
    for (int j = 0; j < n; ++j) if (fscanf(f, " %c", &vt[j]) != 1) return 4;
    int ne;
    if (fscanf(f, "%d", &ne) != 1 || ne < 0) return 4;
    int *ia = calloc((size_t)ne + 1, sizeof(int));
    int *ja = calloc((size_t)ne + 1, sizeof(int));
    double *ar = calloc((size_t)ne + 1, sizeof(double));
    for (int k = 1; k <= ne; ++k) {
        if (fscanf(f, "%d %d", &ia[k], &ja[k]) != 2) return 4;
        ar[k] = read_double(f);
        if (ia[k] < 1 || ia[k] > m || ja[k] < 1 || ja[k] > n) return 4;
    }
    fclose(f);

    glp_term_out(GLP_OFF);
    glp_prob *p = glp_create_prob();
    glp_set_obj_dir(p, maximize ? GLP_MAX : GLP_MIN);
    glp_add_rows(p, m); glp_add_cols(p, n);
    for (int i = 1; i <= m; ++i) {
        double b = rhs[i - 1];
        switch (ct[i - 1]) {
            case 'F': glp_set_row_bnds(p, i, GLP_FR, 0.0, 0.0); break;
            case 'U': glp_set_row_bnds(p, i, GLP_UP, 0.0, b); break;
            case 'S': glp_set_row_bnds(p, i, GLP_FX, b, b); break;
            case 'L': glp_set_row_bnds(p, i, GLP_LO, b, 0.0); break;
            case 'D': glp_set_row_bnds(p, i, GLP_DB, -b, b); break;
            default: fprintf(stderr, "invalid constraint code\n"); return 4;
        }
    }
    for (int j = 1; j <= n; ++j) {
        double lo = lb[j - 1], hi = ub[j - 1];
        if (vt[j - 1] == 'B') { if (lo < 0.0) lo = 0.0; if (hi > 1.0) hi = 1.0; }
        int typ;
        if (isinf(lo) && lo < 0 && isinf(hi) && hi > 0) typ = GLP_FR;
        else if (isinf(lo) && lo < 0) { typ = GLP_UP; lo = 0.0; }
        else if (isinf(hi) && hi > 0) { typ = GLP_LO; hi = 0.0; }
        else if (lo == hi) typ = GLP_FX;
        else typ = GLP_DB;
        glp_set_col_bnds(p, j, typ, lo, hi);
        glp_set_obj_coef(p, j, obj[j - 1]);
        if (vt[j - 1] == 'I') glp_set_col_kind(p, j, GLP_IV);
        else if (vt[j - 1] == 'B') glp_set_col_kind(p, j, GLP_BV);
        else if (vt[j - 1] != 'C') { fprintf(stderr, "invalid variable code\n"); return 4; }
    }
    glp_load_matrix(p, ne, ia, ja, ar);
    glp_smcp parm; glp_init_smcp(&parm);
    parm.msg_lev = GLP_MSG_OFF; parm.meth = GLP_PRIMAL;
    parm.tol_bnd = 1e-6; parm.presolve = GLP_ON;
    int rc = glp_simplex(p, &parm);
    int status = glp_get_status(p);
    int integral = 0;
    for (int j = 0; j < n; ++j) if (vt[j] != 'C') integral = 1;
    if (rc == 0 && integral && (status == GLP_OPT || status == GLP_FEAS)) {
        glp_iocp ip; glp_init_iocp(&ip); ip.msg_lev = GLP_MSG_OFF; ip.presolve = GLP_ON;
        rc = glp_intopt(p, &ip); status = glp_mip_status(p);
    }
    int solved = (status == GLP_OPT || status == GLP_FEAS);
    printf("{\"return_code\":%d,\"status\":%d,\"objective\":", rc, status);
    if (solved) print_value(integral ? glp_mip_obj_val(p) : glp_get_obj_val(p)); else printf("null");
    printf(",\"x\":[");
    for (int j = 1; j <= n; ++j) { if (j > 1) putchar(','); if (!solved) printf("null"); else print_value(integral ? glp_mip_col_val(p, j) : glp_get_col_prim(p, j)); }
    printf("],\"reducedCosts\":");
    if (integral || !solved) { printf("["); for (int j = 0; j < n; ++j) { if (j) putchar(','); printf("null"); } putchar(']'); }
    else print_vector(p, n, 1, 1);
    printf(",\"shadowPrices\":");
    if (integral || !solved) { printf("["); for (int i = 0; i < m; ++i) { if (i) putchar(','); printf("null"); } putchar(']'); }
    else print_vector(p, m, 0, 1);
    puts("}");
    glp_delete_prob(p); glp_free_env();
    free(obj); free(lb); free(ub); free(rhs); free(ct); free(vt); free(ia); free(ja); free(ar);
    return 0;
}
