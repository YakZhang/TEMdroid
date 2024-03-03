import sys
import json
import logging
import random
from abc import abstractmethod

from .input_event import InputEvent, KeyEvent, IntentEvent, TouchEvent, ManualEvent, SetTextEvent, KillAppEvent, OracleEvent, SetTextEnterEvent,ScrollEvent,LongTouchEvent
from .utg import UTG

from .Migration.match_event import get_match_widget, generate_match_event, add_new_function
from .Migration.get_widget_feature_trainBert import get_widget_from_xml,get_all_text_path,get_feature_embedding_from_csv
import os
import pandas as pd



# Max number of restarts
MAX_NUM_RESTARTS = 5
# Max number of steps outside the app
MAX_NUM_STEPS_OUTSIDE = 5
MAX_NUM_STEPS_OUTSIDE_KILL = 10
# Max number of replay tries
MAX_REPLY_TRIES = 5

# Some input event flags
EVENT_FLAG_STARTED = "+started"
EVENT_FLAG_START_APP = "+start_app"
EVENT_FLAG_STOP_APP = "+stop_app"
EVENT_FLAG_EXPLORE = "+explore"
EVENT_FLAG_NAVIGATE = "+navigate"
EVENT_FLAG_TOUCH = "+touch"

# Policy taxanomy
POLICY_NAIVE_DFS = "dfs_naive"
POLICY_GREEDY_DFS = "dfs_greedy"
POLICY_NAIVE_BFS = "bfs_naive"
POLICY_GREEDY_BFS = "bfs_greedy"
POLICY_REPLAY = "replay"
POLICY_MANUAL = "manual"
POLICY_MONKEY = "monkey"
POLICY_NONE = "none"
POLICY_MEMORY_GUIDED = "memory_guided"  # implemented in input_policy2


class InputInterruptedException(Exception):
    pass


class InputPolicy(object):
    """
    This class is responsible for generating events to stimulate more app behaviour
    It should call AppEventManager.send_event method continuously
    """

    def __init__(self, device, app):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.app = app
        self.action_count = 0
        self.master = None

    def start(self, input_manager):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        self.action_count = 0
        while input_manager.enabled and self.action_count < input_manager.event_count:
            try:
                # # make sure the first event is go to HOME screen
                # # the second event is to start the app
                # if self.action_count == 0 and self.master is None:
                #     event = KeyEvent(name="HOME")
                # elif self.action_count == 1 and self.master is None:
                #     event = IntentEvent(self.app.get_start_intent())
                if self.action_count == 0 and self.master is None:
                    event = KillAppEvent(app=self.app)
                else:
                    event, need_stop = self.generate_event()
                    if need_stop is True:
                        break
                    if event is True:
                        continue
                input_manager.add_event(event)
            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.warning("stop sending events: %s" % e)
                break
            # except RuntimeError as e:
            #     self.logger.warning(e.message)
            #     break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()
                continue
            self.action_count += 1

    @abstractmethod
    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass

    def dumpUTG(self):
        """
        dump the UTG
        """
        self.utg.dumpG()

class NoneInputPolicy(InputPolicy):
    """
    do not send any event
    """

    def __init__(self, device, app):
        super(NoneInputPolicy, self).__init__(device, app)

    def generate_event(self):
        """
        generate an event
        @return:
        """
        return None


class UtgBasedInputPolicy(InputPolicy):
    """
    state-based input policy
    """

    def __init__(self, device, app, random_input):
        super(UtgBasedInputPolicy, self).__init__(device, app)
        self.random_input = random_input
        self.script = None
        self.master = None
        self.script_events = []
        self.last_event = None
        self.last_state = None
        self.current_state = None
        self.utg = UTG(device=device, app=app, random_input=random_input)
        self.script_event_idx = 0
        if self.device.humanoid is not None:
            self.humanoid_view_trees = []
            self.humanoid_events = []

        self.src_event_index = 0

        self.event_num = 0 # the first event do not need to match
        self.before_event = None # the before tgt event, if before_event not None but current = None, then back to the last state and find the top_number -1
        self.back_count = 0
        self.check_time = 0
        self.check_max_time = 1
        self.nonconsistent_check_time = 0

        self.generate_event_state = 0

    def generate_event_and_get_stop_signal(self):
        match_widget, src_csv_path, src_event_id, df_src,match_csv_path,src_app_name,tgt_app_name,function,src_event_index = self.combine_droidbot_with_migration()

        # if matched
        if len(match_widget) > 0:
            event, stop = self.generate_migration_event(match_widget, src_csv_path, src_event_id, df_src)
            return event, stop

        # # if not matched

        # if check_num not excess
        if self.check_time + 1 <= self.check_max_time and self.src_event_index + 1 <= len(df_src)-1:
            df_line = pd.DataFrame(
                {
                    'widget1':src_app_name + "/" + function + "-" + str(src_event_index),
                    'widget2':tgt_app_name + "/:" +function + "-" +"/"'false',
                },index=[1]
            )
            self.frame_to_frame(df_line,match_csv_path)
            self.src_event_index += 1
            self.check_time += 1
            return True, False

        # self.generate_event_state = 2

        elif self.nonconsistent_check_time + 1 <= self.check_max_time:
            if self.nonconsistent_check_time == 0:
                self.src_event_index = self.src_event_index - self.check_time
            self.nonconsistent_check_time += 1
            df_line = pd.DataFrame(
                {
                    'widget1':src_app_name +"/"+function + "-" + str(src_event_index),
                    'widget2':tgt_app_name + "/:" +function + "-" +"/"'non_consistent',
                },index=[1]
            )
            self.frame_to_frame(df_line,match_csv_path)
            return None, False

    def frame_to_frame(self,df_line, csv_path):
        if os.path.exists(csv_path) == False:
            df_line.to_csv(csv_path)
        else:
            df = pd.read_csv(csv_path)
            df_new = pd.concat([df, df_line], axis=0, ignore_index=True)
            df_new.to_csv(csv_path)


    def generate_event(self):
        """
        generate an event
        @return:
        """

        # Get current device state
        self.current_state = self.device.get_current_state()
        if self.current_state is None:
            import time
            time.sleep(5)
            return KeyEvent(name="BACK"), False

        self.__update_utg()

        # update last view trees for humanoid
        if self.device.humanoid is not None:
            self.humanoid_view_trees = self.humanoid_view_trees + [self.current_state.view_tree]
            if len(self.humanoid_view_trees) > 4:
                self.humanoid_view_trees = self.humanoid_view_trees[1:]

        event = None

        # judge the common case or start to combine the TEMdroid
        if self.event_num >= 1 \
            and self.device.tgt_xml_prefix is not None \
            and self.device.src_csv_prefix is not None \
            and self.device.src_csv_name is not None:
            print("combine the TEMdroid and droidbot")
            event, stop = self.generate_event_and_get_stop_signal()
            if event is not None and event is not True:
                return event, stop
            if event is True:
                return event, stop

        # if the previous operation is not finished, continue
        if len(self.script_events) > self.script_event_idx:
            event = self.script_events[self.script_event_idx].get_transformed_event(self)
            self.script_event_idx += 1

        # First try matching a state defined in the script
        if event is None and self.script is not None:
            operation = self.script.get_operation_based_on_state(self.current_state)
            if operation is not None:
                self.script_events = operation.events
                # restart script
                event = self.script_events[0].get_transformed_event(self)
                self.script_event_idx = 1

        if event is None:
            event = self.generate_event_based_on_utg()

        # update last events for humanoid
        if self.device.humanoid is not None:
            self.humanoid_events = self.humanoid_events + [event]
            if len(self.humanoid_events) > 3:
                self.humanoid_events = self.humanoid_events[1:]

        self.last_state = self.current_state
        self.last_event = event
        self.event_num += 1
        return event, False

    def __update_utg(self):
        self.utg.add_transition(self.last_event, self.last_state, self.current_state)

    def combine_droidbot_with_migration(self):

        tgt_xml_prefix = self.device.tgt_xml_prefix
        tgt_screen_file = ''
        id = -1
        src_csv_prefix = self.device.src_csv_prefix
        src_csv_name = self.device.src_csv_name
        feature_save_path = self.device.src_tgt_pair_path
        src_event_id = self.device.src_csv_name.split("_")[4].replace(".csv", "") + "-" + str(
            self.src_event_index)  # e.g., 'b1-0'
        print("src_id", src_event_id)
        predict_result_save_path = feature_save_path.replace("test_pair/", "model_new_env/craftdroid/")

        df_src = pd.read_csv(src_csv_prefix + src_csv_name)
        src_app_name = src_csv_name.split("_")[0]
        function = src_csv_name.split("_")[-1].replace(".csv", "")
        tgt_app_name = tgt_xml_prefix.split("/")[-2]
        src_event_index = src_event_id.split("-")[1]

        # for sys_event:
        src_feature_modify_map = get_feature_embedding_from_csv(src_csv_prefix, src_csv_name, src_event_id)
        if src_feature_modify_map == 'SYS_EVENT':
            event = KeyEvent(name='BACK')

            df_line = pd.DataFrame(
                {
                    'widget1': src_app_name + "/" + function + "-"+ src_event_index,
                    'widget2': tgt_app_name + "/:" + function + "/" 'SYS_EVENT',
                    'text_feature1': "SYS_EVENT",
                    'text_feature2': "SYS_EVENT",
                }, index=[1]
            )
            src_app_name = src_csv_name.split("_")[0]
            function = src_csv_name.split("_")[-1].replace(".csv", "")
            tgt_app_name = tgt_xml_prefix.split("/")[-2]
            match_csv_path = predict_result_save_path + src_app_name + "_" + tgt_app_name + "_" + function + "_" + 'test_data_output_predict_map.csv'

            if os.path.exists(match_csv_path) == False:
                df_line.to_csv(match_csv_path)
            else:
                df = pd.read_csv(match_csv_path)
                df_new = pd.concat([df, df_line], axis=0, ignore_index=True)
                df_new.to_csv(match_csv_path)

            if len(df_src) == self.src_event_index + 1:
                return event, True
            else:
                self.src_event_index += 1
                return event, False

        src_tgt_save_path = get_widget_from_xml(tgt_xml_prefix, tgt_screen_file, id, src_csv_prefix, src_csv_name,
                                                feature_save_path, src_event_id)
        src_tgt_save_server_path = src_tgt_save_path.replace("/Users/Migration/",
                                                             'a.b/')


        src_tgt_result_server_path = src_tgt_save_server_path.replace(".jsonl", "") + "_output.csv"
        match_csv_path = predict_result_save_path + src_tgt_result_server_path.split("/")[-1].replace(".csv",
                                                                                                      "") + "_predict_map.csv"


        scp_str = "scp %s website:" + src_tgt_result_server_path.replace(
            src_tgt_result_server_path.split("/")[-1], "")

        os.system(scp_str % src_tgt_save_path)


        tgt_app_name = tgt_xml_prefix.split("/")[-2]  # a13
        corresponding_shell_name = "run1.sh"
        os.system("ssh website \"%s %s\"" % (
        corresponding_shell_name, src_tgt_save_server_path))


        os.system("scp website:%s %s" % (src_tgt_result_server_path, predict_result_save_path))



        threhold = self.device.predict_threhold
        new_function_threahold = self.device.new_function_threhold
        predict_map_path = predict_result_save_path + src_tgt_result_server_path.split("/")[-1]

        src_csv_path = src_csv_prefix + src_csv_name


        add_new_function(predict_map_path, new_function_threahold, predict_map_save_path=predict_map_path)
        match_widget,_ = get_match_widget(threhold, predict_map_path, match_csv_path, back_count=self.back_count)
        return match_widget,src_csv_path,src_event_id,df_src, match_csv_path,src_app_name,tgt_app_name,function,src_event_index

    def generate_migration_event(self,match_widget,src_csv_path,src_event_id,df_src):
            [center_x, center_y, droidbot_type, droidbot_parameter] = generate_match_event(match_widget, src_csv_path,
                                                                                           src_event_id)
            event = ''
            if droidbot_type == 'touch':
                event = TouchEvent(center_x, center_y)
            elif droidbot_type == 'set_text_and_enter':
                event = SetTextEnterEvent(x=center_x, y=center_y, text=droidbot_parameter)
            elif droidbot_type == 'set_text':
                event = SetTextEvent(x=center_x, y=center_y, text=droidbot_parameter)
            elif droidbot_type == 'scroll':
                event = ScrollEvent(center_x, center_y, droidbot_parameter)
            elif droidbot_type == 'long_touch':
                event = LongTouchEvent(center_x, center_y)
            elif droidbot_type == 'oracle':
                event = OracleEvent(center_x, center_y)
            if len(df_src) == self.src_event_index + 1:
                self.before_event = event
                return event, True
            else:
                self.event_num += 1
                self.src_event_index += 1
                self.before_event = event
                return event, False



    @abstractmethod
    def generate_event_based_on_utg(self):
        """
        generate an event based on UTG
        :return: InputEvent
        """
        pass


    def save_to_file(self,xml_data,save_path):
        """
        save the current state hierachy into the output_dir path
        :param xml_data:
        :param save_path:
        :return:
        """
        fh = open(save_path,'w+')
        fh.write(xml_data)
        fh.close()


class UtgNaiveSearchPolicy(UtgBasedInputPolicy):
    """
    depth-first strategy to explore UFG (old)
    """

    def __init__(self, device, app, random_input, search_method):
        super(UtgNaiveSearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.explored_views = set()
        self.state_transitions = set()
        self.search_method = search_method

        self.last_event_flag = ""
        self.last_event_str = None
        self.last_state = None

        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]

    def generate_event_based_on_utg(self):
        """
        generate an event based on current device state
        note: ensure these fields are properly maintained in each transaction:
          last_event_flag, last_touched_view, last_state, exploited_views, state_transitions
        @return: InputEvent
        """
        self.save_state_transition(self.last_event_str, self.last_state, self.current_state)

        if self.device.is_foreground(self.app):
            # the app is in foreground, clear last_event_flag
            self.last_event_flag = EVENT_FLAG_STARTED
        else:
            number_of_starts = self.last_event_flag.count(EVENT_FLAG_START_APP)
            # If we have tried too many times but the app is still not started, stop DroidBot
            if number_of_starts > MAX_NUM_RESTARTS:
                raise InputInterruptedException("The app cannot be started.")

            # if app is not started, try start it
            if self.last_event_flag.endswith(EVENT_FLAG_START_APP):
                # It seems the app stuck at some state, and cannot be started
                # just pass to let viewclient deal with this case
                self.logger.info("The app had been restarted %d times.", number_of_starts)
                self.logger.info("Trying to restart app...")
                pass
            else:
                start_app_intent = self.app.get_start_intent()

                self.last_event_flag += EVENT_FLAG_START_APP
                self.last_event_str = EVENT_FLAG_START_APP
                return IntentEvent(start_app_intent)

        # select a view to click
        view_to_touch = self.select_a_view(self.current_state)

        # if no view can be selected, restart the app
        if view_to_touch is None:
            stop_app_intent = self.app.get_stop_intent()
            self.last_event_flag += EVENT_FLAG_STOP_APP
            self.last_event_str = EVENT_FLAG_STOP_APP
            return IntentEvent(stop_app_intent)

        view_to_touch_str = view_to_touch['view_str']
        if view_to_touch_str.startswith('BACK'):
            result = KeyEvent('BACK')
        else:
            result = TouchEvent(view=view_to_touch)

        self.last_event_flag += EVENT_FLAG_TOUCH
        self.last_event_str = view_to_touch_str
        self.save_explored_view(self.current_state, self.last_event_str)
        return result

    def select_a_view(self, state):
        """
        select a view in the view list of given state, let droidbot touch it
        @param state: DeviceState
        @return:
        """
        views = []
        for view in state.views:
            if view['enabled'] and len(view['children']) == 0:
                views.append(view)

        if self.random_input:
            random.shuffle(views)

        # add a "BACK" view, consider go back first/last according to search policy
        mock_view_back = {'view_str': 'BACK_%s' % state.foreground_activity,
                          'text': 'BACK_%s' % state.foreground_activity}
        if self.search_method == POLICY_NAIVE_DFS:
            views.append(mock_view_back)
        elif self.search_method == POLICY_NAIVE_BFS:
            views.insert(0, mock_view_back)

        # first try to find a preferable view
        for view in views:
            view_text = view['text'] if view['text'] is not None else ''
            view_text = view_text.lower().strip()
            if view_text in self.preferred_buttons \
                    and (state.foreground_activity, view['view_str']) not in self.explored_views:
                self.logger.info("selected an preferred view: %s" % view['view_str'])
                return view

        # try to find a un-clicked view
        for view in views:
            if (state.foreground_activity, view['view_str']) not in self.explored_views:
                self.logger.info("selected an un-clicked view: %s" % view['view_str'])
                return view

        # if all enabled views have been clicked, try jump to another activity by clicking one of state transitions
        if self.random_input:
            random.shuffle(views)
        transition_views = {transition[0] for transition in self.state_transitions}
        for view in views:
            if view['view_str'] in transition_views:
                self.logger.info("selected a transition view: %s" % view['view_str'])
                return view


        # DroidBot stuck on current state, return None
        self.logger.info("no view could be selected in state: %s" % state.tag)
        return None

    def save_state_transition(self, event_str, old_state, new_state):
        """
        save the state transition
        @param event_str: str, representing the event cause the transition
        @param old_state: DeviceState
        @param new_state: DeviceState
        @return:
        """
        if event_str is None or old_state is None or new_state is None:
            return
        if new_state.is_different_from(old_state):
            self.state_transitions.add((event_str, old_state.tag, new_state.tag))

    def save_explored_view(self, state, view_str):
        """
        save the explored view
        @param state: DeviceState, where the view located
        @param view_str: str, representing a view
        @return:
        """
        if not state:
            return
        state_activity = state.foreground_activity
        self.explored_views.add((state_activity, view_str))


class UtgGreedySearchPolicy(UtgBasedInputPolicy):
    """
    DFS/BFS (according to search_method) strategy to explore UFG (new)
    """

    def __init__(self, device, app, random_input, search_method):
        super(UtgGreedySearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_method = search_method

        self.preferred_buttons = ["yes", "ok", "activate", "detail", "more", "access",
                                  "allow", "check", "agree", "try", "go", "next"]

        self.__nav_target = None
        self.__nav_num_steps = -1
        self.__num_restarts = 0
        self.__num_steps_outside = 0
        self.__event_trace = ""
        self.__missed_states = set()
        self.__random_explore = False

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.current_state
        self.logger.info("Current state: %s" % current_state.state_str)
        if current_state.state_str in self.__missed_states:
            self.__missed_states.remove(current_state.state_str)

        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()


            if self.__event_trace.endswith(EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP) \
                    or self.__event_trace.endswith(EVENT_FLAG_START_APP):
                self.__num_restarts += 1
                self.logger.info("The app had been restarted %d times.", self.__num_restarts)
            else:
                self.__num_restarts = 0

            # pass (START) through
            if not self.__event_trace.endswith(EVENT_FLAG_START_APP):
                if self.__num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                    self.__random_explore = True
                else:
                    # Start the app
                    self.__event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    return IntentEvent(intent=start_app_intent)

        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self.__num_steps_outside += 1

            if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self.__event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            # If the app is in foreground
            self.__num_steps_outside = 0

        # Get all possible input events
        if self.device.dfs_graph=='False':
            possible_events = current_state.get_possible_input()
        else:
            # get dfs by category
            possible_events = current_state.get_possible_input_by_category()

        if self.random_input:
            random.shuffle(possible_events)

        if self.search_method == POLICY_GREEDY_DFS:
            possible_events.append(KeyEvent(name="BACK"))
        elif self.search_method == POLICY_GREEDY_BFS:

            possible_events.insert(0, KeyEvent(name="BACK"))

        # get humanoid result, use the result to sort possible events
        # including back events
        if self.device.humanoid is not None:
            possible_events = self.__sort_inputs_by_humanoid(possible_events)

        # If there is an unexplored event, try the event first
        for input_event in possible_events:
            if not self.utg.is_event_explored(event=input_event, state=current_state):
                self.logger.info("Trying an unexplored event.")
                self.__event_trace += EVENT_FLAG_EXPLORE
                return input_event

        target_state = self.__get_nav_target(current_state)
        if target_state:
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=target_state)
            if navigation_steps and len(navigation_steps) > 0:
                self.logger.info("Navigating to %s, %d steps left." % (target_state.state_str, len(navigation_steps)))
                self.__event_trace += EVENT_FLAG_NAVIGATE
                return navigation_steps[0][1]

        if self.__random_explore:
            self.logger.info("Trying random event.")
            random.shuffle(possible_events)
            return possible_events[0]

        # If couldn't find a exploration target, stop the app
        stop_app_intent = self.app.get_stop_intent()
        self.logger.info("Cannot find an exploration target. Trying to restart app...")
        self.__event_trace += EVENT_FLAG_STOP_APP
        return IntentEvent(intent=stop_app_intent)

    def __sort_inputs_by_humanoid(self, possible_events):
        if sys.version.startswith("3"):
            from xmlrpc.client import ServerProxy
        else:
            from xmlrpclib import ServerProxy
        proxy = ServerProxy("http://%s/" % self.device.humanoid)
        request_json = {
            "history_view_trees": self.humanoid_view_trees,
            "history_events": [x.__dict__ for x in self.humanoid_events],
            "possible_events": [x.__dict__ for x in possible_events],
            "screen_res": [self.device.display_info["width"],
                           self.device.display_info["height"]]
        }
        result = json.loads(proxy.predict(json.dumps(request_json)))
        new_idx = result["indices"]
        text = result["text"]
        new_events = []

        # get rid of infinite recursive by randomizing first event
        if not self.utg.is_state_reached(self.current_state):
            new_first = random.randint(0, len(new_idx) - 1)
            new_idx[0], new_idx[new_first] = new_idx[new_first], new_idx[0]

        for idx in new_idx:
            if isinstance(possible_events[idx], SetTextEvent):
                possible_events[idx].text = text
            new_events.append(possible_events[idx])
        return new_events

    def __get_nav_target(self, current_state):
        # If last event is a navigation event
        if self.__nav_target and self.__event_trace.endswith(EVENT_FLAG_NAVIGATE):
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if navigation_steps and 0 < len(navigation_steps) <= self.__nav_num_steps:
                # If last navigation was successful, use current nav target
                self.__nav_num_steps = len(navigation_steps)
                return self.__nav_target
            else:
                # If last navigation was failed, add nav target to missing states
                self.__missed_states.add(self.__nav_target.state_str)

        reachable_states = self.utg.get_reachable_states(current_state)
        if self.random_input:
            random.shuffle(reachable_states)

        for state in reachable_states:
            # Only consider foreground states
            if state.get_app_activity_depth(self.app) != 0:
                continue
            # Do not consider missed states
            if state.state_str in self.__missed_states:
                continue
            # Do not consider explored states
            if self.utg.is_state_explored(state):
                continue
            self.__nav_target = state
            navigation_steps = self.utg.get_navigation_steps(from_state=current_state, to_state=self.__nav_target)
            if len(navigation_steps) > 0:
                self.__nav_num_steps = len(navigation_steps)
                return state

        self.__nav_target = None
        self.__nav_num_steps = -1
        return None

class UtgReplayPolicy(InputPolicy):
    """
    Replay DroidBot output generated by UTG policy
    """

    def __init__(self, device, app, replay_output):
        super(UtgReplayPolicy, self).__init__(device, app)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.replay_output = replay_output

        import os
        event_dir = os.path.join(replay_output, "events")
        self.event_paths = sorted([os.path.join(event_dir, x) for x in
                                   next(os.walk(event_dir))[2]
                                   if x.endswith(".json")])
        # skip HOME and start app intent
        self.event_idx = 2
        self.num_replay_tries = 0

    def generate_event(self):
        """
        generate an event based on replay_output
        @return: InputEvent
        """
        import time
        while self.event_idx < len(self.event_paths) and \
              self.num_replay_tries < MAX_REPLY_TRIES:
            self.num_replay_tries += 1
            current_state = self.device.get_current_state()
            if current_state is None:
                time.sleep(5)
                self.num_replay_tries = 0
                return KeyEvent(name="BACK")

            curr_event_idx = self.event_idx
            while curr_event_idx < len(self.event_paths):
                event_path = self.event_paths[curr_event_idx]
                with open(event_path, "r") as f:
                    curr_event_idx += 1

                    try:
                        event_dict = json.load(f)
                    except Exception as e:
                        self.logger.info("Loading %s failed" % event_path)
                        continue

                    if event_dict["start_state"] != current_state.state_str:
                        continue

                    self.logger.info("Replaying %s" % event_path)
                    self.event_idx = curr_event_idx
                    self.num_replay_tries = 0
                    return InputEvent.from_dict(event_dict["event"])

            time.sleep(5)

        raise InputInterruptedException("No more record can be replayed.")


class ManualPolicy(UtgBasedInputPolicy):
    """
    manually explore UFG
    """

    def __init__(self, device, app):
        super(ManualPolicy, self).__init__(device, app, False)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.__first_event = True

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        if self.__first_event:
            self.__first_event = False
            self.logger.info("Trying to start the app...")
            start_app_intent = self.app.get_start_intent()
            return IntentEvent(intent=start_app_intent)
        else:
            return ManualEvent()
