#ifndef __COMMON_RLWM_KERNEL_H__
#define __COMMON_RLWM_KERNEL_H__
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

/**
 * Device kernel for rlwm_compute
 */
#include <cmath>
#include <algorithm>

template<typename T>
__device__ T solve(const T u, const T y, const T guess, const T nu,
                   const T kappa, const T B);

template<typename T>
__device__ void calculate_reward(const int i, const T tau_new, const T tau_true, const T tau_old,
                                T * __restrict__ error_new_d, T * __restrict__ error_old_d, 
                                T * __restrict__ rel_error_d, T * __restrict__ reward_d,
                                T * __restrict__ total_reward_d, T * __restrict__ base_reward_d, 
                                T * __restrict__ bonus_reward_d);

/**
 * CUDA kernel for Spalding.
 */
template<typename T>
__global__ void spalding_initialize(const T * __restrict__ u_d, const T * __restrict__ v_d, const T * __restrict__ w_d,
                             const int * __restrict__ ind_r_d, const int * __restrict__ ind_s_d, 
                             const int * __restrict__ ind_t_d, const int * __restrict__ ind_e_d,
                             const T * __restrict__ n_x_d, const T * __restrict__ n_y_d, const T * __restrict__ n_z_d,
                             const T * __restrict__ nu_d, const T * __restrict__ h_d,
                             T * __restrict__ tau_x_d, T * __restrict__ tau_y_d, T * __restrict__ tau_z_d,
                             const int n_nodes, const int lx, const T kappa, const T B,
                             const int tstep, const T tau_true, 
                             T * __restrict__ ui_l_d, T * __restrict__ vi_l_d, T * __restrict__ wi_l_d,
                             T * __restrict__ normu_l_d, T * __restrict__ magu_l_d,
                             T * __restrict__ vg_l_d, T * __restrict__ utau_l_d,
                             T * __restrict__ tau_old_l_d, T * __restrict__ tau_new_l_d, T * __restrict__ l_star_d,
                             T * __restrict__ u_plus_d, T * __restrict__ g_plus_d, T * __restrict__ h_plus_d, 
                             T * __restrict__ slope_d, T * __restrict__ intercept_d, 
                             const T * __restrict__ dudy_d,
                             T * __restrict__ error_new_d, T * __restrict__ error_old_d,
                             T * __restrict__ rel_error_d, T * __restrict__ reward_d,
                             T * __restrict__ total_reward_d, const T * __restrict__ reward_out_d,
                             T * __restrict__ base_reward_d, T * __restrict__ bonus_reward_d,
                             T * __restrict__ terminal_d,
                             T * __restrict__ state_d, T * __restrict__ action_d, 
                             const int * __restrict__ msk_d, T * __restrict__ reward_field_d, 
                             T * __restrict__ slope_field_d, T * __restrict__ intercept_field_d) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int str = blockDim.x * gridDim.x;
    for (int i = idx; i < n_nodes; i += str) {
        // Sample the velocity
        const int index = (ind_e_d[i] - 1) * lx * lx * lx +
                          (ind_t_d[i] - 1) * lx * lx +
                          (ind_s_d[i] - 1) * lx +
                          (ind_r_d[i] - 1);

        T ui = u_d[index];
        T vi = v_d[index];
        T wi = w_d[index];

        // Load normal vectors and wall shear stress values once
        T nx = n_x_d[i];
        T ny = n_y_d[i];
        T nz = n_z_d[i];
        T h = h_d[i];

        // Project on tangential direction
        T normu = ui * nx + vi * ny + wi * nz;

        ui -= normu * nx;
        vi -= normu * ny;
        wi -= normu * nz;

        T magu = sqrt(ui * ui + vi * vi + wi * wi);

        ui_l_d[i] = ui;
        vi_l_d[i] = vi;
        wi_l_d[i] = wi;
        normu_l_d[i] = normu;
        magu_l_d[i] = magu;
        
        // Get initial guess for Newton solver
        T guess;
        if (tstep == 1) {

            // Guess velocity
            vg_l_d[i] = sqrt(magu_l_d[i] * nu_d[i] / h_d[i]);

            // Friction velocity
            utau_l_d[i] = solve(magu, h, vg_l_d[i], nu_d[i], kappa, B);

            // Shear Stress resulting from the computed utau
            tau_old_l_d[i] = utau_l_d[i] * utau_l_d[i];

            // Distribute according to the velocity vector
            tau_x_d[i] = -utau_l_d[i] * utau_l_d[i] * ui_l_d[i] / magu_l_d[i];
            tau_y_d[i] = -utau_l_d[i] * utau_l_d[i] * vi_l_d[i] / magu_l_d[i];
            tau_z_d[i] = -utau_l_d[i] * utau_l_d[i] * wi_l_d[i] / magu_l_d[i];

        } else {

            // Magnitude of Shear Stress
            tau_old_l_d[i] = sqrt(tau_x_d[i] * tau_x_d[i] + tau_y_d[i] * tau_y_d[i] + tau_z_d[i] * tau_z_d[i]);

            // Guess Velocity
            vg_l_d[i] = sqrt(tau_old_l_d[i]);

            // Friction velocity
            utau_l_d[i] = solve(magu, h, vg_l_d[i], nu_d[i], kappa, B);

            // Shear Stress resulting from the computed utau
            tau_new_l_d[i] = utau_l_d[i] * utau_l_d[i];

            // Distribute according to the velocity vector
            tau_x_d[i] = -utau_l_d[i] * utau_l_d[i] * ui_l_d[i] / magu_l_d[i];
            tau_y_d[i] = -utau_l_d[i] * utau_l_d[i] * vi_l_d[i] / magu_l_d[i];
            tau_z_d[i] = -utau_l_d[i] * utau_l_d[i] * wi_l_d[i] / magu_l_d[i];

            if (i==1){
                printf("COMPUTE: %d, tau_new_l_d[%d]=%.15f, tau_old_l_d[%d]=%.15f\n", 
                    tstep, i, tau_new_l_d[i], i, tau_old_l_d[i]);
            }
            
            // Calculate reward
            calculate_reward(i, tau_new_l_d[i], tau_true, tau_old_l_d[i],
                           error_new_d, error_old_d, rel_error_d, reward_d,
                           total_reward_d, base_reward_d, bonus_reward_d);

            // Copy reward value to the field for visualization
            reward_field_d[msk_d[i]-1] = reward_d[i];

        }

       // Normalization w.r.t. viscous scales (nu, utau)
       l_star_d[i] = nu_d[i] / utau_l_d[i]; // l_star_d[i] = nu_d[i] / (utau_l_d[i] + 1e-6);
       u_plus_d[i] = magu_l_d[i] / utau_l_d[i]; // u_plus_d[i] = magu_l_d[i] / (utau_l_d[i] + 1e-6);
       g_plus_d[i] = dudy_d[index] / ( utau_l_d[i] / l_star_d[i]); // g_plus_d[i] = dudy_d[index] / ( (utau_l_d[i] + 1e-6) / l_star_d[i]);
       h_plus_d[i] = h_d[i] / l_star_d[i]; // h_plus_d[i] = h_d[i] / (l_star_d[i] + 1e-6);

       // Changing to normalized states
	   slope_d[i] = h_plus_d[i] * g_plus_d[i] - (log(h_plus_d[i])/kappa);
	   intercept_d[i] = u_plus_d[i] - (log(h_plus_d[i])/kappa);

	   // RL State 
	   state_d[0 * n_nodes + i] = slope_d[i];      // First feature
	   state_d[1 * n_nodes + i] = intercept_d[i];  // Second feature
	   
	   // Field for visualization
	   slope_field_d[msk_d[i]-1] = slope_d[i];
	   intercept_field_d[msk_d[i]-1] = intercept_d[i];
    }
}

/**
 * CUDA kernel for RLWM.
 */
template<typename T>
__global__ void rlwm_compute(const T * __restrict__ u_d, const T * __restrict__ v_d, const T * __restrict__ w_d,
                             const int * __restrict__ ind_r_d, const int * __restrict__ ind_s_d, 
                             const int * __restrict__ ind_t_d, const int * __restrict__ ind_e_d,
                             const T * __restrict__ n_x_d, const T * __restrict__ n_y_d, const T * __restrict__ n_z_d,
                             const T * __restrict__ nu_d, const T * __restrict__ h_d,
                             T * __restrict__ tau_x_d, T * __restrict__ tau_y_d, T * __restrict__ tau_z_d,
                             const int n_nodes, const int lx, const T kappa, const T B,
                             const int tstep, const int model_device, const int rb_device, const int start_rl_tstep,
                             const int tsteps_rl, const int episode_length, const int n_epochs, const T tau_true, 
                             T * __restrict__ ui_l_d, T * __restrict__ vi_l_d, T * __restrict__ wi_l_d,
                             T * __restrict__ normu_l_d, T * __restrict__ magu_l_d,
                             T * __restrict__ vg_l_d, T * __restrict__ utau_l_d,
                             T * __restrict__ tau_old_l_d, T * __restrict__ tau_new_l_d, T * __restrict__ l_star_d,
                             T * __restrict__ u_plus_d, T * __restrict__ g_plus_d, T * __restrict__ h_plus_d, 
                             T * __restrict__ slope_d, T * __restrict__ intercept_d, 
                             const T * __restrict__ dudy_d,
                             T * __restrict__ error_new_d, T * __restrict__ error_old_d,
                             T * __restrict__ rel_error_d, T * __restrict__ reward_d,
                             T * __restrict__ total_reward_d, const T * __restrict__ reward_out_d,
                             T * __restrict__ base_reward_d, T * __restrict__ bonus_reward_d,
                             T * __restrict__ terminal_d,
                             T * __restrict__ state_d, T * __restrict__ action_d, 
                             const int * __restrict__ msk_d, T * __restrict__ reward_field_d, 
                             T * __restrict__ slope_field_d, T * __restrict__ intercept_field_d) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int str = blockDim.x * gridDim.x;
    for (int i = idx; i < n_nodes; i += str) {
        // Sample the velocity
        const int index = (ind_e_d[i] - 1) * lx * lx * lx +
                          (ind_t_d[i] - 1) * lx * lx +
                          (ind_s_d[i] - 1) * lx +
                          (ind_r_d[i] - 1);

        T ui = u_d[index];
        T vi = v_d[index];
        T wi = w_d[index];

        // Load normal vectors and wall shear stress values once
        T nx = n_x_d[i];
        T ny = n_y_d[i];
        T nz = n_z_d[i];
        T h = h_d[i];

        // Project on tangential direction
        T normu = ui * nx + vi * ny + wi * nz;

        ui -= normu * nx;
        vi -= normu * ny;
        wi -= normu * nz;

        T magu = sqrt(ui * ui + vi * vi + wi * wi);

        ui_l_d[i] = ui;
        vi_l_d[i] = vi;
        wi_l_d[i] = wi;
        normu_l_d[i] = normu;
        magu_l_d[i] = magu;
        
        if ((tstep - start_rl_tstep) % tsteps_rl == 0) {
            // Only then calculate episode_step and set terminal flag
            int rl_step = (tstep - start_rl_tstep) / tsteps_rl;
            int episode_step = rl_step % episode_length;
            if (episode_step == 0 && rl_step > 0) {
                terminal_d[i] = 1.0;
            } else {
                terminal_d[i] = 0.0;
            }
        } else {
            // For non-RL timesteps, preserve previous terminal value or set to 0
            terminal_d[i] = 0.0;
        }
        // First set the terminal flag for the current timestep 
        // if (((tstep - start_rl_tstep) / tsteps_rl) % episode_length == 0) {
        //   terminal_d[i] = 1.0;
        // } else {
        //   terminal_d[i] = 0.0;
        // }

        // Magnitude of Shear Stress
        tau_old_l_d[i] = sqrt(tau_x_d[i] * tau_x_d[i] + tau_y_d[i] * tau_y_d[i] + tau_z_d[i] * tau_z_d[i]);

        // Friction velocity
        utau_l_d[i] = sqrt(tau_old_l_d[i]);

        // Guess Velocity
        vg_l_d[i] = utau_l_d[i];

        if (i==1){
            printf("RLWM-COMPUTE: %d, tau_new[%d]=%.5f, tau_old[%d]=%.5f\n", 
                tstep, i, tau_new_l_d[i], i, tau_old_l_d[i]);
        }

        // Normalization w.r.t. viscous scales (nu, utau)
        l_star_d[i] = nu_d[i] / utau_l_d[i]; // l_star_d[i] = nu_d[i] / (utau_l_d[i] + 1e-6);
        u_plus_d[i] = magu_l_d[i] / utau_l_d[i]; // u_plus_d[i] = magu_l_d[i] / (utau_l_d[i] + 1e-6);
        g_plus_d[i] = dudy_d[index] / ( utau_l_d[i] / l_star_d[i]); // g_plus_d[i] = dudy_d[index] / ( (utau_l_d[i] + 1e-6) / l_star_d[i]);
        h_plus_d[i] = h_d[i] / l_star_d[i]; // h_plus_d[i] = h_d[i] / (l_star_d[i] + 1e-6);

        // Changing to normalized states
        slope_d[i] = h_plus_d[i] * g_plus_d[i] - (log(h_plus_d[i])/kappa);
        intercept_d[i] = u_plus_d[i] - (log(h_plus_d[i])/kappa);

        // RL State 
        state_d[0 * n_nodes + i] = slope_d[i];      // First feature
        state_d[1 * n_nodes + i] = intercept_d[i];  // Second feature
        
        // Field for visualization
        slope_field_d[msk_d[i]-1] = slope_d[i];
        intercept_field_d[msk_d[i]-1] = intercept_d[i];
    }
}

/**
 * Kernel for applying the actions to the wall shear stress
 */
template<typename T>
__global__ void rlwm_actuate(const int n_nodes, const int tstep, 
                             const int start_rl_tstep, const int tsteps_rl, const int episode_length,
                             const T * __restrict__ action_d,
                             T * __restrict__ tau_old_l_d, T * __restrict__ tau_new_l_d,
                             T * __restrict__ utau_l_d,
                             T * __restrict__ tau_x_d, T * __restrict__ tau_y_d, T * __restrict__ tau_z_d,
                             const T tau_true,
                             T * __restrict__ ui_l_d, T * __restrict__ vi_l_d, T * __restrict__ wi_l_d,
                             T * __restrict__ magu_l_d,
                             T * __restrict__ error_new_d, T * __restrict__ error_old_d,
                             T * __restrict__ rel_error_d, T * __restrict__ reward_d,
                             T * __restrict__ total_reward_d, 
                             T * __restrict__ base_reward_d, T * __restrict__ bonus_reward_d,
                             const int * __restrict__ msk_d, T * __restrict__ reward_field_d) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int str = blockDim.x * gridDim.x;
    for (int i = idx; i < n_nodes; i += str) {
        if (((tstep - start_rl_tstep) % tsteps_rl) == 0) {
            // Store the RL-predicted tau for under-relaxation
            tau_new_l_d[i] = tau_old_l_d[i] * action_d[i];
            // Calculate reward
            calculate_reward(i, tau_new_l_d[i], tau_true, tau_old_l_d[i],
                            error_new_d, error_old_d, rel_error_d, reward_d,
                            total_reward_d, base_reward_d, bonus_reward_d);
            if (i==1){
                printf("NEW-ACTION: %d, tau_new[%d]=%.5f, tau_old[%d]=%.5f, action_d[%d]=%.5f\n", 
                    tstep, i, tau_new_l_d[i], i, tau_old_l_d[i], i, action_d[i]);
            }
        } else {
            // Under-relax between physics-computed tau (tau_old_l_d) and RL-predicted tau (tau_new_l_d from last action)
            // Template T to avoid 0 due to integer division truncation
            T alpha = T((tstep - start_rl_tstep) % tsteps_rl) / T(tsteps_rl);
            tau_new_l_d[i] = tau_old_l_d[i] * (T(1.0) - alpha) + tau_new_l_d[i] * alpha;
            if (i==1){
                printf("UNDER-RELAX: %d, tau_new[%d]=%.5f, tau_old[%d]=%.5f, alpha=%.5f\n", 
                    tstep, i, tau_new_l_d[i], i, tau_old_l_d[i], alpha);
            }
        }

        // Friction velocity based on new wall shear stress
        utau_l_d[i] = sqrt(tau_new_l_d[i]);

        // Distribute according to the velocity vector
        tau_x_d[i] = -utau_l_d[i] * utau_l_d[i] * ui_l_d[i] / magu_l_d[i];
        tau_y_d[i] = -utau_l_d[i] * utau_l_d[i] * vi_l_d[i] / magu_l_d[i];
        tau_z_d[i] = -utau_l_d[i] * utau_l_d[i] * wi_l_d[i] / magu_l_d[i];
        
        // // Calculate reward
        // calculate_reward(i, tau_new_l_d[i], tau_true, tau_old_l_d[i],
        //                 error_new_d, error_old_d, rel_error_d, reward_d,
        //                 total_reward_d, base_reward_d, bonus_reward_d);

        // Copy reward value to the field for visualization
        reward_field_d[msk_d[i]-1] = reward_d[i];
    }
}


/**
 * Kernel for under-relaxing the wall shear stress
 */
template<typename T>
__global__ void rlwm_under_relax(const int n_nodes, const int tstep, 
                                 const int start_rl_tstep, const int tsteps_rl, const int episode_length,
                                 const T * __restrict__ action_d,
                                 T * __restrict__ tau_old_l_d, T * __restrict__ tau_new_l_d,
                                 T * __restrict__ utau_l_d,
                                 T * __restrict__ tau_x_d, T * __restrict__ tau_y_d, T * __restrict__ tau_z_d,
                                 const T tau_true,
                                 T * __restrict__ ui_l_d, T * __restrict__ vi_l_d, T * __restrict__ wi_l_d,
                                 T * __restrict__ magu_l_d,
                                 T * __restrict__ error_new_d, T * __restrict__ error_old_d,
                                 T * __restrict__ rel_error_d, T * __restrict__ reward_d,
                                 T * __restrict__ total_reward_d, 
                                 T * __restrict__ base_reward_d, T * __restrict__ bonus_reward_d,
                                 const int * __restrict__ msk_d, T * __restrict__ reward_field_d) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int str = blockDim.x * gridDim.x;
    for (int i = idx; i < n_nodes; i += str) {
        // Under-relax between physics-computed tau (tau_old_l_d) and RL-predicted tau (tau_new_l_d from last action)
        // Template T to avoid 0 due to integer division truncation
        T alpha = T((tstep - start_rl_tstep) % tsteps_rl) / T(tsteps_rl);
        tau_new_l_d[i] = tau_old_l_d[i] * (T(1.0) - alpha) + tau_new_l_d[i] * alpha;
        if (i==1){
        printf("UNDER-RELAX: %d, tau_new[%d]=%.5f, tau_old[%d]=%.5f, alpha=%.5f\n", 
            tstep, i, tau_new_l_d[i], i, tau_old_l_d[i], alpha);
        }
        // Friction velocity based on new wall shear stress
        utau_l_d[i] = sqrt(tau_new_l_d[i]);

        // Distribute according to the velocity vector
        tau_x_d[i] = -utau_l_d[i] * utau_l_d[i] * ui_l_d[i] / magu_l_d[i];
        tau_y_d[i] = -utau_l_d[i] * utau_l_d[i] * vi_l_d[i] / magu_l_d[i];
        tau_z_d[i] = -utau_l_d[i] * utau_l_d[i] * wi_l_d[i] / magu_l_d[i];
        
        // // Calculate reward
        // calculate_reward(i, tau_new_l_d[i], tau_true, tau_old_l_d[i],
        //                 error_new_d, error_old_d, rel_error_d, reward_d,
        //                 total_reward_d, base_reward_d, bonus_reward_d);

        // Copy reward value to the field for visualization
        reward_field_d[msk_d[i]-1] = reward_d[i];        
    }
}

/**
 * Kernel for applying the actions to the wall shear stress
 */
template<typename T>
__global__ void rlwm_inference(const int n_nodes, const int tstep, 
                               const int start_rl_tstep, const int tsteps_rl, const int episode_length,
                               const T * __restrict__ action_d,
                               T * __restrict__ tau_old_l_d, T * __restrict__ tau_new_l_d,
                               T * __restrict__ utau_l_d,
                               T * __restrict__ tau_x_d, T * __restrict__ tau_y_d, T * __restrict__ tau_z_d,
                               const T tau_true,
                               T * __restrict__ ui_l_d, T * __restrict__ vi_l_d, T * __restrict__ wi_l_d,
                               T * __restrict__ magu_l_d,
                               T * __restrict__ error_new_d, T * __restrict__ error_old_d,
                               T * __restrict__ rel_error_d, T * __restrict__ reward_d,
                               T * __restrict__ total_reward_d, 
                               T * __restrict__ base_reward_d, T * __restrict__ bonus_reward_d,
                               const int * __restrict__ msk_d, T * __restrict__ reward_field_d) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int str = blockDim.x * gridDim.x;

    for (int i = idx; i < n_nodes; i += str) {

        if (((tstep - start_rl_tstep) % tsteps_rl) == 0) {
            tau_new_l_d[i] = tau_old_l_d[i] * action_d[i];
            if (i==1){
                printf("INFERENCE-NEW-ACTION: %d, tau_new[%d]=%.5f, tau_old[%d]=%.5f, action_d[%d]=%.5f\n", 
                    tstep, i, tau_new_l_d[i], i, tau_old_l_d[i], i, action_d[i]);
            }
        } else {
            T alpha = T((tstep - start_rl_tstep) % tsteps_rl) / T(tsteps_rl);
            tau_new_l_d[i] = tau_old_l_d[i] * (T(1.0) - alpha) + tau_new_l_d[i] * alpha;
            if (i==1){
                printf("INFERENCE-UNDER-RELAX: %d, tau_new[%d]=%.5f, tau_old[%d]=%.5f, alpha=%.5f\n", 
                    tstep, i, tau_new_l_d[i], i, tau_old_l_d[i], alpha);
            }
        }

        // Friction velocity based on new wall shear stress
        utau_l_d[i] = sqrt(tau_new_l_d[i]);

        // Distribute according to the velocity vector
        tau_x_d[i] = -utau_l_d[i] * utau_l_d[i] * ui_l_d[i] / magu_l_d[i];
        tau_y_d[i] = -utau_l_d[i] * utau_l_d[i] * vi_l_d[i] / magu_l_d[i];
        tau_z_d[i] = -utau_l_d[i] * utau_l_d[i] * wi_l_d[i] / magu_l_d[i];
    }
}

/**
 * Newton solver for the algebraic equation defined by the law on GPU.
 */
template<typename T>
__device__ T solve(const T u, const T y, const T guess, const T nu,
                   const T kappa, const T B) {
    T utau = guess;
    T yp, up, f, df, old, error;
    const int maxiter = 100;

    for (int k = 0; k < maxiter; ++k) {
        up = u / utau;
        yp = y * utau / nu;
        old = utau;

        // Evaluate function and its derivative
        f = (up + exp(-kappa * B) *
                  (exp(kappa * up) - 1.0 - kappa * up -
                   0.5 * (kappa * up) * (kappa * up) -
                   (1.0 / 6.0) * (kappa * up) * (kappa * up) * (kappa * up)) -
             yp);

        df = (-y / nu - u / (utau * utau) -
              kappa * up / utau * exp(-kappa * B) *
                  (exp(kappa * up) - 1.0 - kappa * up -
                   0.5 * (kappa * up) * (kappa * up)));

        // Update solution
        utau -= f / df;

        error = fabs((old - utau) / old);

        if (error < 1e-3) {
            break;
        }
    }

    return utau;
}

/**
 * Calculate reward for RLWM on GPU.
 */
template<typename T>
__device__ void calculate_reward(const int i, const T tau_new, const T tau_true, const T tau_old,
                                T * __restrict__ error_new_d, T * __restrict__ error_old_d, 
                                T * __restrict__ rel_error_d, T * __restrict__ reward_d,
                                T * __restrict__ total_reward_d, T * __restrict__ base_reward_d, 
                                T * __restrict__ bonus_reward_d) {
    
    // Method 1: Base + Bonus reward system 
    error_new_d[i] = fabs(tau_true - tau_new);
    error_old_d[i] = fabs(tau_true - tau_old);
    base_reward_d[i] = (error_new_d[i] - error_old_d[i]) / tau_true;
    
    rel_error_d[i] = error_new_d[i] / tau_true;
    if (rel_error_d[i] < T(0.01)) {
        bonus_reward_d[i] = T(1.0) - rel_error_d[i];
    } else {
        bonus_reward_d[i] = T(0.0);
    }
    
    // Reward collected by agent 'i' at current timestep
    reward_d[i] = base_reward_d[i] + bonus_reward_d[i];

    // Reward clipping
    reward_d[i] = tanh(reward_d[i]);
    
    // Reward collected by agent 'i' in one episode consisting of trajectories at 'tsteps_rl' 
    total_reward_d[i] = total_reward_d[i] + reward_d[i];
    
    // Method 2: Simple relative error
    // reward_d[i] = fabs(tau_new - tau_true) / tau_true;
    // total_reward_d[i] = total_reward_d[i] + reward_d[i];
}

#endif // __COMMON_RLWM_KERNEL_H__