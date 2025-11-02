/*
 Copyright (c) 2025, The Neko Authors
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.

   * Neither the name of the authors nor the names of its
     contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <device/device_config.h>
#include <device/cuda/check.h>
#include "rlwm_kernel.h"

extern "C" {
  void cuda_spalding_initialize(void *u_d, void *v_d, void *w_d,
          void *ind_r_d, void *ind_s_d, void *ind_t_d, void *ind_e_d,
          void *n_x_d, void *n_y_d, void *n_z_d, void *nu_d, void *h_d,
          void *tau_x_d, void *tau_y_d, void *tau_z_d,
          int *n_nodes, int *lx, real *kappa, real *B, int *tstep,
          real *tau_true,
          void *ui_l_d, void *vi_l_d, void *wi_l_d, void *normu_l_d, void *magu_l_d, void *vg_l_d, void *utau_l_d,
          void *tau_old_l_d, void *tau_new_l_d,
          void *l_star_d, void *u_plus_d, void *g_plus_d, void *h_plus_d, void *slope_d, void *intercept_d, void *dudy_d,
          void *error_new_d, void *error_old_d, void *rel_error_d,
          void *reward_d, void *total_reward_d, void *reward_out_d, void *base_reward_d, void *bonus_reward_d,
          void *terminal_d,
          void *state_d, void *action_d, 
          void *msk_d, void *reward_field_d, void *slope_field_d, void *intercept_field_d) {

    const dim3 nthrds(512, 1, 1);
    const dim3 nblcks(((*n_nodes)+512 - 1)/ 512, 1, 1);
    const cudaStream_t stream = (cudaStream_t) glb_cmd_queue;

    spalding_initialize<real>
    <<<nblcks, nthrds, 0, stream>>>((real *) u_d, (real *) v_d, (real *) w_d,
                                    (int *) ind_r_d, (int *) ind_s_d, (int *) ind_t_d, (int *) ind_e_d,
                                    (real *) n_x_d, (real *) n_y_d, (real *) n_z_d, 
                                    (real* ) nu_d, (real *) h_d,
                                    (real *) tau_x_d, (real *) tau_y_d, (real *) tau_z_d,
                                    *n_nodes, *lx, *kappa, *B, *tstep,
                                    *tau_true,
                                    (real *) ui_l_d, (real *) vi_l_d, (real *) wi_l_d, (real *) normu_l_d, 
                                    (real *) magu_l_d, (real *) vg_l_d, (real *) utau_l_d,
                                    (real *) tau_old_l_d, (real *) tau_new_l_d,
                                    (real *) l_star_d, (real *) u_plus_d, (real *) g_plus_d, (real *) h_plus_d, 
                                    (real *) slope_d, (real *) intercept_d, (real *) dudy_d,
                                    (real *) error_new_d, (real *) error_old_d, (real *) rel_error_d,
                                    (real *) reward_d, (real *) total_reward_d, (real *) reward_out_d, 
                                    (real *) base_reward_d, (real *) bonus_reward_d,
                                    (real *) terminal_d,
                                    (real *) state_d, (real *) action_d, 
                                    (int *) msk_d, (real *) reward_field_d, (real *) slope_field_d, (real *) intercept_field_d);
    CUDA_CHECK(cudaGetLastError());
  }

  void cuda_rlwm_compute(void *u_d, void *v_d, void *w_d,
          void *ind_r_d, void *ind_s_d, void *ind_t_d, void *ind_e_d,
          void *n_x_d, void *n_y_d, void *n_z_d, void *nu_d, void *h_d,
          void *tau_x_d, void *tau_y_d, void *tau_z_d,
          int *n_nodes, int *lx, real *kappa, real *B, int *tstep,
          int *model_device, int *rb_device, int *start_rl_tstep, int *tsteps_rl, int *episode_length, int *n_epochs, real *tau_true,
          void *ui_l_d, void *vi_l_d, void *wi_l_d, void *normu_l_d, void *magu_l_d, void *vg_l_d, void *utau_l_d,
          void *tau_old_l_d, void *tau_new_l_d,
          void *l_star_d, void *u_plus_d, void *g_plus_d, void *h_plus_d, void *slope_d, void *intercept_d, void *dudy_d,
          void *error_new_d, void *error_old_d, void *rel_error_d,
          void *reward_d, void *total_reward_d, void *reward_out_d, void *base_reward_d, void *bonus_reward_d,
          void *terminal_d,
          void *state_d, void *action_d, 
          void *msk_d, void *reward_field_d, void *slope_field_d, void *intercept_field_d) {

    const dim3 nthrds(512, 1, 1);
    const dim3 nblcks(((*n_nodes)+512 - 1)/ 512, 1, 1);
    const cudaStream_t stream = (cudaStream_t) glb_cmd_queue;

    rlwm_compute<real>
    <<<nblcks, nthrds, 0, stream>>>((real *) u_d, (real *) v_d, (real *) w_d,
                                    (int *) ind_r_d, (int *) ind_s_d, (int *) ind_t_d, (int *) ind_e_d,
                                    (real *) n_x_d, (real *) n_y_d, (real *) n_z_d, 
                                    (real* ) nu_d, (real *) h_d,
                                    (real *) tau_x_d, (real *) tau_y_d, (real *) tau_z_d,
                                    *n_nodes, *lx, *kappa, *B, *tstep,
                                    *model_device, *rb_device, *start_rl_tstep, *tsteps_rl, *episode_length, *n_epochs, *tau_true,
                                    (real *) ui_l_d, (real *) vi_l_d, (real *) wi_l_d, (real *) normu_l_d, 
                                    (real *) magu_l_d, (real *) vg_l_d, (real *) utau_l_d,
                                    (real *) tau_old_l_d, (real *) tau_new_l_d,
                                    (real *) l_star_d, (real *) u_plus_d, (real *) g_plus_d, (real *) h_plus_d, 
                                    (real *) slope_d, (real *) intercept_d, (real *) dudy_d,
                                    (real *) error_new_d, (real *) error_old_d, (real *) rel_error_d,
                                    (real *) reward_d, (real *) total_reward_d, (real *) reward_out_d, 
                                    (real *) base_reward_d, (real *) bonus_reward_d,
                                    (real *) terminal_d,
                                    (real *) state_d, (real *) action_d, 
                                    (int *) msk_d, (real *) reward_field_d, (real *) slope_field_d, (real *) intercept_field_d);
    CUDA_CHECK(cudaGetLastError());
  }

  void cuda_rlwm_actuate(int *n_nodes, int *tstep, int *start_rl_tstep, int *tsteps_rl, int *episode_length,
          void *action_d,
          void *tau_old_l_d, void *tau_new_l_d, void *utau_l_d,
          void *tau_x_d, void *tau_y_d, void *tau_z_d,
          real *tau_true,
          void *ui_l_d, void *vi_l_d, void *wi_l_d, void *magu_l_d,
          void *error_new_d, void *error_old_d, void *rel_error_d,
          void *reward_d, void *total_reward_d,
          void *base_reward_d, void *bonus_reward_d,
          void *msk_d, void *reward_field_d) {

    const dim3 nthrds(512, 1, 1);
    const dim3 nblcks(((*n_nodes)+512 - 1)/ 512, 1, 1);
    const cudaStream_t stream = (cudaStream_t) glb_cmd_queue;

    rlwm_actuate<real>
    <<<nblcks, nthrds, 0, stream>>>(*n_nodes, *tstep, *start_rl_tstep, *tsteps_rl, *episode_length,
                                    (real *) action_d,
                                    (real *) tau_old_l_d, (real *) tau_new_l_d, (real *) utau_l_d,
                                    (real *) tau_x_d, (real *) tau_y_d, (real *) tau_z_d,
                                    *tau_true,
                                    (real *) ui_l_d, (real *) vi_l_d, (real *) wi_l_d, (real *) magu_l_d,
                                    (real *) error_new_d, (real *) error_old_d, (real *) rel_error_d,
                                    (real *) reward_d, (real *) total_reward_d,
                                    (real *) base_reward_d, (real *) bonus_reward_d, 
                                    (int *) msk_d, (real *) reward_field_d);
    CUDA_CHECK(cudaGetLastError());
  }

  void cuda_rlwm_under_relax(int *n_nodes, int *tstep, int *start_rl_tstep, int *tsteps_rl, int *episode_length,
          void *action_d,
          void *tau_old_l_d, void *tau_new_l_d, void *utau_l_d,
          void *tau_x_d, void *tau_y_d, void *tau_z_d,
          real *tau_true,
          void *ui_l_d, void *vi_l_d, void *wi_l_d, void *magu_l_d,
          void *error_new_d, void *error_old_d, void *rel_error_d,
          void *reward_d, void *total_reward_d,
          void *base_reward_d, void *bonus_reward_d,
          void *msk_d, void *reward_field_d) {

    const dim3 nthrds(512, 1, 1);
    const dim3 nblcks(((*n_nodes)+512 - 1)/ 512, 1, 1);
    const cudaStream_t stream = (cudaStream_t) glb_cmd_queue;

    rlwm_under_relax<real>
    <<<nblcks, nthrds, 0, stream>>>(*n_nodes, *tstep, *start_rl_tstep, *tsteps_rl, *episode_length,
                                    (real *) action_d,
                                    (real *) tau_old_l_d, (real *) tau_new_l_d, (real *) utau_l_d,
                                    (real *) tau_x_d, (real *) tau_y_d, (real *) tau_z_d,
                                    *tau_true,
                                    (real *) ui_l_d, (real *) vi_l_d, (real *) wi_l_d, (real *) magu_l_d,
                                    (real *) error_new_d, (real *) error_old_d, (real *) rel_error_d,
                                    (real *) reward_d, (real *) total_reward_d,
                                    (real *) base_reward_d, (real *) bonus_reward_d, 
                                    (int *) msk_d, (real *) reward_field_d);
    CUDA_CHECK(cudaGetLastError());
  }

}