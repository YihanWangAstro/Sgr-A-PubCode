#include <algorithm>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/numeric/odeint.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
const double G = 6.674e-8;
const double c = 2.99792458e10;
const double ms = 1.989e33;
const double yr = 3.15576e7;
std::vector<double> BH_pop3;
std::vector<double> BH_clusters;
std::vector<double> BH_direct;

using Array = std::vector<double>;
template <typename T>
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

struct Disk {
    double alpha{0.1};
    double Q{1};
    double Edd_ratio{1};
    double eta{0.1};
    double lx{0};
    double ly{0};
    double lz{1};
    double BZ_eff{0};
};

inline static double Uniform(double low, double high) {
    static thread_local std::mt19937 generator{std::random_device{}()};
    std::uniform_real_distribution<double> dist{low, high};
    return dist(generator);
}

inline static double Logarithm(double low, double high) {
    static thread_local std::mt19937 generator{std::random_device{}()};
    double log_low = log10(low);
    double log_high = log10(high);
    std::uniform_real_distribution<double> dist{log_low, log_high};
    return pow(10, dist(generator));
}

inline static double PowerLaw(double power, double low, double high) {
    static thread_local std::mt19937 generator{std::random_device{}()};
    if (std::fabs(power + 1.0) > 1e-6) {
        double beta = power + 1;
        double f_low = pow(low, beta);
        double f_high = pow(high, beta);
        std::uniform_real_distribution<double> dist{f_low, f_high};
        return pow(dist(generator), 1.0 / beta);
    } else {
        return Logarithm(low, high);
    }
}

double t_sg(double M, Disk disk) {
    return 1.12e6 * pow(disk.alpha / 0.03, -2.0 / 27) * pow(disk.eta / 0.1, 22.0 / 27) * pow(M / 1e8 / ms, -4.0 / 27) *
           yr;
}

struct Accfunc {
    Accfunc(Disk disk) : _disk{disk} {}

    void operator()(Array& x, Array& dxdt, double t) {
        double ax = x[0];
        double ay = x[1];
        double az = x[2];
        double M = x[3];
        double a = sqrt(ax * ax + ay * ay + az * az);
        if (a >= 0.9999) {
            x[0] = ax / a;
            x[1] = ay / a;
            x[2] = az / a;
        }
        double jet_x = 0;
        double jet_y = 0;
        double jet_z = 0;

        if (a >= 1e-6) {
            jet_x = ax / a;
            jet_y = ay / a;
            jet_z = az / a;
        }

        double M8 = (M / (1e8 * ms));
        double Z1 = 1 + pow(1 - a * a, 1.0 / 3) * (pow(1 + a, 1.0 / 3) + pow(1 - a, 1.0 / 3));
        double Z2 = sqrt(3 * a * a + Z1 * Z1);
        double a_dot_L = ax * _disk.lx + ay * _disk.ly + az * _disk.lz;
        double Rms = (3 + Z2 - sgn(a_dot_L) * sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))) / 2;
        double Ems = (4 * sqrt(Rms) - 3 * a) / sqrt(3) / Rms;
        double Lms = G * M / c * (6 * sqrt(Rms) - 4 * a) / sqrt(3) / Rms;

        double Mdot_edd = 2.2 * M8 * ms / yr;
        double Mdot = Mdot_edd * _disk.Edd_ratio;
        double Omega_f = 0.5 * c * c * c / G / M * a / (1 + sqrt(1 - a * a)) / 2;
        double P_jet = 2.5 * (a / (1 + sqrt(1 - a * a))) * (a / (1 + sqrt(1 - a * a))) * _disk.BZ_eff * _disk.BZ_eff *
                       Mdot * c * c;
        double L_jet = P_jet / Omega_f;
        double dMdt = Mdot * Ems - P_jet / (c * c);

        if (dMdt < 0) {
            dMdt = 0;
        }

        double daxdt = (Mdot * Lms * _disk.lx - L_jet * jet_x) * c / (G * M * M) - 2 * ax * dMdt / M;
        double daydt = (Mdot * Lms * _disk.ly - L_jet * jet_y) * c / (G * M * M) - 2 * ay * dMdt / M;
        double dazdt = (Mdot * Lms * _disk.lz - L_jet * jet_z) * c / (G * M * M) - 2 * az * dMdt / M;
        dxdt[0] = daxdt;
        dxdt[1] = daydt;
        dxdt[2] = dazdt;
        dxdt[3] = dMdt;
    }
    Disk _disk;
};

bool accretion_cycle(Disk const& disk, Array& x, double t_acc, double M_max, std::ofstream& file, double& time,
                     int& cycles) {
    using namespace boost::numeric::odeint;
    double atol = 1e-13;
    double rtol = 1e-13;

    auto stepper = bulirsch_stoer<std::vector<double>>{atol, rtol};
    auto func = Accfunc{disk};
    double time0 = time;
    double dt = yr / 1000;
    double dt_out = t_acc / 100;
    double time_out = time;

    double a = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);

    file << time / yr << ' ' << a << ' ' << x[2] / a << ' ' << x[3] / ms << ' ' << cycles << '\n';

    cycles++;

    for (; time <= time0 + t_acc;) {
        if (time >= time_out) {
            double a = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            file << time / yr << ' ' << a << ' ' << x[2] / a << ' ' << x[3] / ms << ' ' << cycles << '\n';
            time_out += dt_out;
        }
        constexpr size_t max_attempts = 500;
        controlled_step_result res = success;
        size_t trials = 0;
        do {
            res = stepper.try_step(func, x, time, dt);
            trials++;
        } while ((res == fail) && (trials < max_attempts));

        if (trials == max_attempts) {
            std::cout << "cannot get converged results\n";
            exit(0);
        }
        double M_now = x[3];
        if (M_now > M_max) {
            return true;
        }
    }
    return false;
}

auto gen_disk_orientation(int iso_k) {
    if (iso_k > 30) {
        return std::make_tuple(0.0, 0.0, 1.0);
    }
    const double pi = 3.14159265358979323846;
    double pdf_max = exp(iso_k * 1) / 2 / pi / boost::math::cyl_bessel_i(0, iso_k);
    for (;;) {
        double cosi = Uniform(-1, 1);
        double p = Uniform(0, pdf_max);
        double pdf = exp(iso_k * cosi) / 2 / pi / boost::math::cyl_bessel_i(0, iso_k);
        if (p < pdf) {
            double phi = Uniform(0, 2 * pi);
            double sini = sqrt(1 - cosi * cosi);
            return std::make_tuple(sini * cos(phi), sini * sin(phi), cosi);
        }
    }
}

auto iso_orien() {
    double const pi = 3.14159265358979323846;
    double cosi = Uniform(-1, 1);
    double phi = Uniform(0, 2 * pi);
    double sini = sqrt(1 - cosi * cosi);
    return std::make_tuple(sini * cos(phi), sini * sin(phi), cosi);
}

void accretion_traj(std::string seed, double a_idx, double M_max, double Edd, double dc_ratio, double BZ_eff,
                    int iso_k) {
    char fname[200] = {0};
    sprintf(fname, "paper-run/traj-%s-%.2lf-%.2lf-%.2lf-%.2lf-%d.txt", seed.c_str(), a_idx, Edd, dc_ratio, BZ_eff,
            iso_k);
    std::ofstream file(fname);

    double M0 = 10;
    if (seed == "pop3") {
        M0 = BH_pop3[std::rand() % BH_pop3.size()];
    } else if (seed == "clusters") {
        M0 = BH_clusters[std::rand() % BH_clusters.size()];
    } else if (seed == "direct") {
        M0 = BH_direct[std::rand() % BH_direct.size()];
    }
    double a0 = a_idx;

    auto [nax, nay, naz] = std::make_tuple(0.0, 0.0, 1.0);  // iso_orien();
    Array x0 = {a0 * nax, a0 * nay, a0 * naz, M0 * ms};

    double time = 0;
    int acc_num = 0;
    // file << time / yr << ' ' << M0 << ' ' << a0 << ' ' << a0 << ' ' << naz << ' ' << M0 << ' ' << acc_num << '\n';
    for (; time < 1.3e10 * yr;) {
        auto [lx, ly, lz] = gen_disk_orientation(iso_k);
        Disk disk{0.1, 1, Edd, 0.1, lx, ly, lz, BZ_eff};
        double M_now = x0[3];
        double t_acc = t_sg(M_now, disk);
        bool is_caped = accretion_cycle(disk, x0, t_acc, M_max, file, time, acc_num);

        if (!is_caped) {
            time += t_acc * (1 / dc_ratio - 1);
        } else {
            return;
        }

        /*double ax = x0[0];
        double ay = x0[1];
        double az = x0[2];
        double M = x0[3];
        double a = sqrt(ax * ax + ay * ay + az * az);
        double cos_theta = az / a;*/

        /*if (acc_time < t_acc) {
            time += acc_time;
            file << time / yr << ' ' << M0 << ' ' << a0 << ' ' << a << ' ' << cos_theta << ' ' << M / ms << ' '
                 << acc_num << '\n';
            break;
        } else {
            file << (time + t_acc) / yr << ' ' << M0 << ' ' << a0 << ' ' << a << ' ' << cos_theta << ' ' << M / ms
                 << ' ' << acc_num << '\n';
            time += t_acc / dc_ratio;
            if (fabs(dc_ratio - 1) > 1e-6) {
                file << time / yr << ' ' << M0 << ' ' << a0 << ' ' << a << ' ' << cos_theta << ' ' << M / ms << ' '
                     << acc_num << '\n';
            }
        }*/
    }
}

void load_BH_mass() {
    std::ifstream file("BH-pop3.csv");
    double M;
    while (file >> M) BH_pop3.push_back(M);
    file.close();
    file.open("BH-clusters.csv");
    while (file >> M) BH_clusters.push_back(M);
    file.close();
    file.open("BH-direct.csv");
    while (file >> M) BH_direct.push_back(M);
    file.close();
}

int main() {
    load_BH_mass();

    if (BH_pop3.size() == 0 || BH_clusters.size() == 0 || BH_direct.size() == 0) {
        std::cout << "BH mass file not found\n";
        return 0;
    }
    std::vector<std::thread> threads;

    double M_max = 4.1e6 * ms;

    std::vector<std::string> seeds = {"pop3", "clusters", "direct"};

    double as[] = {0.01, 0.99};
    double edds[] = {1, 0.1, 0.01};
    double dc_ratios[] = {0.1, 0.01, 1};
    double BZ_effs[] = {0, 0.1, 1};
    double iso_ks[] = {0, 1, 2, 30, 40};

    for (auto Edd : edds) {
        for (auto dc_ratio : dc_ratios) {
            for (auto BZ_eff : BZ_effs) {
                for (auto iso_k : iso_ks) {
                    for (auto a_init : as) {
                        for (auto M_init_idx : seeds) {
                            threads.emplace_back(
                                std::thread{accretion_traj, M_init_idx, a_init, M_max, Edd, dc_ratio, BZ_eff, iso_k});
                        }
                    }
                }
            }
        }
    }
    for (auto& t : threads) t.join();
}