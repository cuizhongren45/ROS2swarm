#!/usr/bin/env python3
#    Copyright 2020 Marian Begemann
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy
import random

from ros2swarm.utils import setup_node
from ros2swarm.utils.vote_list import VoteList
from ros2swarm.voting_pattern.voting_pattern import VotingPattern
from communication_interfaces.msg import OpinionMACMessage

from ros2swarm.utils.wifi_functions import WifiFunctions


class VoterModelWithLimiterPattern(VotingPattern):
    """
    Implementation of the Voter Model using dBm strength of others as limiter.

    Pattern to reach conclusion by setting the own option to the opinion of an random nearby
    other robot. The nearby range is defined by the wifi strength.
    The opinion could be any integer.

    pattern_node >> communicate under the topic: vote_channel
    """

    def __init__(self):
        """Initialize the voter model pattern node."""
        super().__init__('voter_model_pattern')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('voter_model_with_limiter_number_of_near_robots', None),
                ('voter_model_with_limiter_initial_value', None),
                ('voter_model_with_limiter_choose_start_value_at_random', None),
                ('voter_model_with_limiter_min_opinion', None),
                ('voter_model_with_limiter_max_opinion', None),
                ('voter_model_with_limiter_timer_period', None),
                ('voter_model_with_limiter_required_dDm_to_be_near', None),
                ('voter_model_with_limiter_wifi_interface_name', None),
            ])

        self.param_number_of_near_robots = self.get_parameter(
            "voter_model_with_limiter_number_of_near_robots").get_parameter_value().integer_value
        self.param_initial_value = self.get_parameter(
            "voter_model_with_limiter_initial_value").get_parameter_value().integer_value
        self.param_choose_start_value_at_random = self.get_parameter(
            "voter_model_with_limiter_choose_start_value_at_random") \
            .get_parameter_value().bool_value
        self.param_min_opinion = self.get_parameter(
            "voter_model_with_limiter_min_opinion").get_parameter_value().integer_value
        self.param_max_opinion = self.get_parameter(
            "voter_model_with_limiter_max_opinion").get_parameter_value().integer_value
        param_timer_period = self.get_parameter(
            "voter_model_with_limiter_timer_period").get_parameter_value().double_value
        self.param_required_dBm = self.get_parameter(
            "voter_model_with_limiter_required_dDm_to_be_near").get_parameter_value().integer_value
        self.param_wifi_interface_name = self.get_parameter(
            "voter_model_with_limiter_wifi_interface_name").get_parameter_value().string_value

        # find out own mac address
        self.mac_address = WifiFunctions.get_own_mac_address(self.param_wifi_interface_name)

        # get robot id
        self.id = super.get_robot_id()

        # set initial opinion
        if self.param_choose_start_value_at_random:
            self.opinion = numpy.random.randint(self.param_min_opinion, self.param_max_opinion)
        else:
            self.opinion = self.param_initial_value

        # create reused OpinionMessage
        self.opinion_message = OpinionMACMessage()
        self.opinion_message.id = self.id
        self.opinion_message.opinion = self.opinion
        self.opinion_message.mac = self.mac_address

        # lists to store opinions
        self.opinion_list = []
        self.final_opinion_list = []

        # define time period to listen to other opinions
        self.timer = self.create_timer(param_timer_period, self.swarm_command_controlled_timer(self.timer_callback))

        # OpinionMACMessage: {id[integer],opinion[integer],mac[string]}
        self.broadcast_publisher = self.create_publisher(OpinionMACMessage,
                                                         '/voting_broadcast',
                                                         10)

        self.broadcast_subscription = self.create_subscription(
            OpinionMACMessage,
            '/voting_broadcast',
            self.swarm_command_controlled(self.voting_broadcast_callback),
            10)

        self.first_broadcast_flag = False

    def timer_callback(self):
        """Select a new opinion of another entity and emit the own opinion."""
        self.get_logger().debug('Robot "{}" has opinion "{}" and a list of size "{}"'
                                .format(self.get_namespace(), self.opinion,
                                        len(self.opinion_list)))

        # update opinion if at least one opinion were received and initial opinion send once
        if len(self.opinion_list) > 0 and self.first_broadcast_flag:
            self.get_new_opinion()

        # emit opinion
        self.opinion_message.opinion = self.opinion
        self.broadcast_publisher.publish(self.opinion_message)
        self.first_broadcast_flag = True

    def get_new_opinion(self):
        """Get the new opinion."""
        # freeze list
        self.final_opinion_list = self.opinion_list

        # get station dump
        station_to_dBm = WifiFunctions.get_all_stations_and_strength(
            self.param_wifi_interface_name)

        # sort list by dBm
        station_to_dBm.sort(key=lambda x: x[1])

        # remove all with a too weak signal
        strong_stations = [s for s in station_to_dBm if s[1] < self.param_required_dBm]

        if len(strong_stations) == 0:
            return

        # remove all which where not heard
        heard_macs = [e.mac for e in self.final_opinion_list]
        heard_stations = [s for s in strong_stations if heard_macs.__contains__(s[0])]

        if len(heard_stations) == 0:
            return

        # limit remaining stations to a defined number
        remaining_stations = heard_stations[:self.param_number_of_near_robots]

        if len(remaining_stations) == 0:
            return

        # choose a station by random
        chosen_station = random.choice(remaining_stations)

        # get the opinion of the chosen station
        chosen_opinion = [e for e in self.final_opinion_list if e.mac == chosen_station[0]][0]

        # set new opinion
        self.opinion = chosen_opinion.opinion

        # reset lists
        self.opinion_list = []
        self.final_opinion_list = []

    def voting_broadcast_callback(self, opinion_msg):
        """Store heard opinion message in a list to use it later."""
        self.opinion_list = VoteList.update_opinion(self.opinion_list, opinion_msg, self.id)


def main(args=None):
    """
    Create a node for the voter model with limiter pattern.

    Spins it and handles the destruction.
    """
    setup_node.init_and_spin(args, VoterModelWithLimiterPattern)


if __name__ == '__main__':
    main()
