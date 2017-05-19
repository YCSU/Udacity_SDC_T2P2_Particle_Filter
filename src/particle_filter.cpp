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
#include <limits>

#include "particle_filter.h"

using namespace std;

default_random_engine gen{1};

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_psi(theta, std[2]);
  
    particles.resize(this -> num_particles);
    if( !is_initialized ){
      for( auto  p = particles.begin(); p != particles.end(); ++p){
          p -> x = dist_x(gen);
          p -> y = dist_y(gen);
          p -> theta = dist_psi(gen);
          p -> weight = 1.;
      }
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    normal_distribution<double> std_x(0, std_pos[0]);
    normal_distribution<double> std_y(0, std_pos[1]);
    normal_distribution<double> std_psi(0, std_pos[2]);
    
    if(yaw_rate > 0.001){
        for( auto  p = particles.begin(); p != particles.end(); ++p){
            p -> x = p -> x + velocity / yaw_rate * (sin(p -> theta + yaw_rate * delta_t) - sin(p -> theta)) + std_x(gen);
            p -> y = p -> y + velocity / yaw_rate * (cos(p -> theta) - cos(p -> theta + yaw_rate * delta_t)) + std_y(gen);
            p -> theta = p -> theta + yaw_rate * delta_t + std_psi(gen);
        }
    }else{
        for( auto  p = particles.begin(); p != particles.end(); ++p){
            p -> x = p -> x + velocity * delta_t * cos(p -> theta) + std_x(gen);
            p -> y = p -> y + velocity * delta_t * sin(p -> theta) + std_y(gen);
            p -> theta = p -> theta + yaw_rate * delta_t + std_psi(gen);
        }      
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> &predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for(auto pre = predicted.begin(); pre != predicted.end(); ++pre){
        int idx = 0;
        double min_dist = std::numeric_limits<double>::max();
        for(auto obs = observations.begin(); obs != observations.end(); ++obs){
            double d = dist(pre -> x, pre -> y, obs -> x, obs -> y);
            if(d < min_dist){
                pre -> id = idx;
                min_dist = d;
            }
            ++idx;
        }
        cout << "pre_id: " << pre -> id << endl;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
    for(auto p = this->particles.begin(); p != this->particles.end(); ++p){
        
        std::vector<LandmarkObs> predicted;
        for(auto landmark = map_landmarks.landmark_list.begin(); 
                landmark != map_landmarks.landmark_list.end(); ++landmark){
            if( dist( landmark -> x_f, landmark -> y_f, p -> x, p -> y) < sensor_range){
                double dx = landmark -> x_f - p -> x;
                double dy = landmark -> y_f - p -> y;
                LandmarkObs obs;
                obs.x =  dx * cos(p -> theta) + dy * sin(p -> theta);
                obs.y = -dx * sin(p -> theta) + dy * cos(p -> theta);
                predicted.push_back(obs);    
            }
            //cout << dist( landmark -> x_f, landmark -> y_f, p -> x, p -> y) << endl;
            
            //for(auto obs = observations.begin(); obs != observations.end(); ++obs){
            //  double map_obs_x = p -> x + obs -> x * cos(p -> theta) - obs -> y * sin(p -> theta);
            //  double map_obs_y = p -> y + obs -> x * sin(p -> theta) + obs -> y * cos(p -> theta);
            //}
        }
        cout << "*******************" << predicted.size() << "***********" << observations.size() << endl;
        dataAssociation(predicted, observations);
        
        p -> weight = 1.;
        for(auto pre = predicted.begin(); pre != predicted.end(); ++pre){
            cout << "weight: "<< p-> weight << endl;
            cout << "id: " << pre-> id << endl;
            p -> weight *= 0.5 / std_landmark[0] / std_landmark[1] / M_PI
                          * exp(
                                 -0.5 * (
                                          (pre->x - observations[pre->id].x) * (pre->x - observations[pre->id].x) / std_landmark[0] / std_landmark[0] +
                                          (pre->y - observations[pre->id].y) * (pre->y - observations[pre->id].y) / std_landmark[1] / std_landmark[1]
                                        )
                               );
            
        }
        
    }      
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    this -> weights.clear();
    for(auto p = this -> particles.begin(); p != this -> particles.end(); ++p){
        this -> weights.push_back(p -> weight);
    }
    
    discrete_distribution<int> rand_idx(this -> weights.begin(), this -> weights.end());
    vector<Particle> new_particles;
    for(auto p = this -> particles.begin(); p != this->particles.end(); ++p){
        new_particles.push_back(this -> particles[rand_idx(gen)]);
    }
    
    this -> particles = new_particles;
    
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
