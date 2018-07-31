/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 1000;

  default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

  for(int p=0; p<num_particles; ++p) {
    Particle particle;
    particle.id = p;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  for(int p_id=0; p_id < num_particles; ++p_id) {
    double px = particles[p_id].x;
    double py = particles[p_id].y;
    double ptheta = particles[p_id].theta;

    if(fabs(yaw_rate) < 1e-5) {
      px += velocity*cos(ptheta)*delta_t;
      py += velocity*sin(ptheta)*delta_t;
    } else {
      px += (velocity/yaw_rate)*(sin(ptheta + yaw_rate*delta_t) - sin(ptheta));
      py += (velocity/yaw_rate)*(cos(ptheta) - cos(ptheta + yaw_rate*delta_t));
      ptheta += yaw_rate*delta_t;
    }

    normal_distribution<double> dist_x(px, std_pos[0]);
    normal_distribution<double> dist_y(py, std_pos[1]);
    normal_distribution<double> dist_theta(ptheta, std_pos[2]);

    particles[p_id].x = dist_x(gen);
    particles[p_id].y = dist_y(gen);
    particles[p_id].theta = dist_theta(gen);

  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  vector<LandmarkObs>::iterator obs_it;
  vector<LandmarkObs>::iterator pred_it;
  for(obs_it = observations.begin(); obs_it != observations.end(); ++obs_it) {
    double shortest_dist = numeric_limits<double>::max();
    int shortest_id = -1; //landmark id of the nearest neighbor
    for(pred_it = predicted.begin(); pred_it != predicted.end(); ++pred_it) {
      double distance = dist(obs_it->x, obs_it->y, pred_it->x, pred_it->y);
      if (distance < shortest_dist) {
        shortest_dist = distance;
        shortest_id = pred_it->id;
      }
    }
    //std::cout << "shortest distance = " << shortest_dist << std::endl;
    obs_it->id = shortest_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // goes through the list of all updated particles
  //   makes a vector of all landmarks within the range
  //   transforms car observations to map observations
  //   finds nearest visible neighbor for each observation
  //   calculates multivariate normal distribution and updates particle's weight

  for(int id=0; id < particles.size(); ++id) {
    double xp = particles[id].x;
    double yp = particles[id].y;
    double theta = particles[id].theta;

    //compose a list of visible landmarks
    vector<LandmarkObs> visible_landmarks;
    for(int i=0; i < map_landmarks.landmark_list.size(); ++i) {
      double xm = map_landmarks.landmark_list[i].x_f;
      double ym = map_landmarks.landmark_list[i].y_f;
      int id_m = map_landmarks.landmark_list[i].id_i;
      if (dist(xp, yp, xm, ym) < sensor_range) {
        LandmarkObs vis_landmark;
        vis_landmark.x = xm;
        vis_landmark.y = ym;
        vis_landmark.id = id_m;
        visible_landmarks.push_back(vis_landmark);
      }
    }

    //find the closest landmark to each observation
    vector<LandmarkObs> my_obs(observations);
    vector<LandmarkObs>::iterator obs_it;
    for(obs_it = my_obs.begin(); obs_it != my_obs.end(); ++obs_it) {
      double xc = obs_it->x;
      double yc = obs_it->y;
      //int   idc = obs_it->id;
      obs_it->x = cos(theta)*xc - sin(theta)*yc + xp;
      obs_it->y = sin(theta)*xc + cos(theta)*yc + yp;
    }

    //nearest neighbor within sensor's range
    dataAssociation(visible_landmarks, my_obs);

    double gauss_norm = 1./(2*M_PI*std_landmark[0]*std_landmark[1]);
    double cum_dx2 = 0.0;
    double cum_dy2 = 0.0;
    //double weight = 1.0;

    for(obs_it = my_obs.begin(); obs_it != my_obs.end(); ++obs_it) {
      double xc = obs_it->x;
      double yc = obs_it->y;
      int   idc = obs_it->id;

      double xl = map_landmarks.landmark_list[idc-1].x_f;
      double yl = map_landmarks.landmark_list[idc-1].y_f;

      //double gauss_power = (xc - xl)*(xc - xl)/(2*std_landmark[0]*std_landmark[0]);
      //gauss_power += (yc - yl)*(yc - yl)/(2*std_landmark[1]*std_landmark[1]);
      //weight *= gauss_norm * exp(-gauss_power);

      double dx2 = (xc - xl)*(xc - xl);
      double dy2 = (yc - yl)*(yc - yl);

      cum_dx2 += dx2;
      cum_dy2 += dy2;
    }

    // update particle's weight:
    // adds all exponents divide by the variances
    // normalizes by the gaussian norm multiplied num_particles times
    // saves a couple of seconds over mulitiplying normalized exponents
    cum_dx2 /= std_landmark[0]*std_landmark[0];
    cum_dy2 /= std_landmark[1]*std_landmark[1];
    particles[id].weight = pow(gauss_norm, my_obs.size()) * exp(-0.5*(cum_dx2 + cum_dy2));
    //particles[id].weight = weight;

  }


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<double> all_weights;
  for(int id=0; id < particles.size(); ++id) {
    all_weights.push_back(particles[id].weight);
  }

  default_random_engine gen;
  //weight normalization is not necessary. It is weighed automatically.
  discrete_distribution<> d(all_weights.begin(), all_weights.end());

  //resamples particles based on their weight
  vector<Particle> new_particles;
  for(int i=0; i < particles.size(); ++i) {
    new_particles.push_back(particles[d(gen)]);
  }
  particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
