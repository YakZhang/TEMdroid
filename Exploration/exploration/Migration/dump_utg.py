import os
import json

def get_png_name(state_map, state_id):
    json_filename = state_map[state_id]
    png_filename = json_filename.replace("state","xml").replace(".json",".xml")
    return png_filename

def get_app_name(js_file_path):
    app_name = js_file_path.split("/")[-2]
    return app_name


def __key_if_true(view_dict, key):
    return view_dict[key] if (key in view_dict and view_dict[key]) else ""


def get_view_name(state_file_path, state_map, state_id, view_id):

    file = state_map[state_id]
    with open(state_file_path + file, 'r') as f:
        state_data = json.load(f)
        for view in state_data['views']:
            if view['view_str'] == view_id:
                resource_id = __key_if_true(view, 'resource_id')
                if 'id/' in resource_id:
                    resource_id = resource_id.split("id/")[1]

                text = __key_if_true(view, 'text')
                content_description = __key_if_true(view, 'content_description')
                return '%s#%s#%s' % (resource_id, text, content_description)


def generate_json_file(js_file_path, js_utg_new_file_path):
    if os.path.exists(js_utg_new_file_path)==True:
        return js_utg_new_file_path
    app_name = get_app_name(js_file_path)
    if '/utg.js' not in js_file_path:
        js_file_path += '/utg.js'
    json_file_path_prefix = js_file_path.replace(js_file_path.split("/")[-1], "")
    json_file_path = json_file_path_prefix + "utg.json"
    with open(js_file_path, mode='r', encoding='utf-8') as f:
        line = f.readlines()
        try:
            line = line[1:]
            f = open(json_file_path, mode='w', encoding='utf-8')
            f.writelines(line)
            f.close()
        except:
            pass

    with open(json_file_path, 'r') as utg_file:
        utg_data = json.load(utg_file)

    for node in utg_data['nodes']:
        node['name'] = app_name+"/"+node['image'].replace("screen","xml").replace(".png",".xml")

    state_id_str_dict = {}
    states_file_path = json_file_path_prefix+ 'states/'
    for root,dirs,files in os.walk(states_file_path):
        for file in files:
            if '.json' in file:
                with open(states_file_path + file, 'r') as f:
                    state_data = json.load(f)
                    key = state_data['state_str']
                    val = file
                    state_id_str_dict[key] = val

    for i in range(len(utg_data['edges'])):
        edge = utg_data['edges'][i]
        if len(edge['events']) == 0:
            continue
        if len(edge['events']) == 1:
            continue
        for j in range(1, len(edge['events'])):
            utg_data['edges'].append(edge)
            new_edge = utg_data['edges'][-1]
            if j <= len(new_edge['events'])-1:
                new_edge['events'] = [new_edge['events'][j]]
        edge['events'] = [edge['events'][0]]


    removed_edge = []
    for edge in utg_data['edges']:
        A = app_name+"/states/"+get_png_name(state_id_str_dict, edge['from'])
        assert edge['events'] is not None
        if len(edge['events']) > 1:
            print("more than one events")
        # assert len(edge['events']) == 1
        if len(edge['events']) == 0: # events=[]
            removed_edge.append(edge)
            continue
        if len(edge['events'][0]['view_images']) == 0:
            B = edge['events'][0]['event_type']
        else:
            view_id = edge['events'][0]['event_str'].split('view=')[1].split('(')[0]
            B = get_view_name(states_file_path, state_id_str_dict, edge['from'], view_id)
        C = app_name+"/states/"+get_png_name(state_id_str_dict, edge['to'])
        edge['name'] = '%s:%s:%s' % (A, B, C)
        edge['from_state_name'] = A
        edge['widget_name'] = B
        edge['to_state_name'] = C

    for edge in removed_edge:
        utg_data['edges'].remove(edge)

    launch_node = None
    launch_node_name = None
    for node in utg_data['nodes']:
        if 'Launcher' in node['activity']:
            launch_node = node
            launch_node_name = node['name']
            break
    utg_data['nodes'].remove(launch_node)

    launch_edge = None
    for edge in utg_data['edges']:
        if edge['from_state_name'] == launch_node_name:
            launch_edge = edge
            break
    if launch_edge != None:
        utg_data['edges'].remove(launch_edge)



    json_file_new_path = json_file_path_prefix+"utg-new.json"
    with open(json_file_new_path, "w") as dump_f:
        json.dump(utg_data, dump_f, indent=2)
    print("json_file_new_path",json_file_new_path)
    return json_file_new_path

def query_edge(utg_file, str_a, str_b):
    if isinstance(utg_file, str):
        with open(json_file_path, 'r') as utg_file:
            utg_data = json.load(utg_file)
    else:
        utg_data = utg_file

    if utg_data is None or str_a is None or str_b is None:
        return None

    str_b = str_b.replace("nan","")

    tgt_state = None
    tgt_state_name = None
    for edge in utg_data['edges']:
        if edge['from_state_name'] == str_a and edge['widget_name'] == str_b:
            tgt_state_name = edge['to_state_name']
            break
    if tgt_state_name == None:
        print("cannot find a to_tgt_state")
        return None

    for node in utg_data['nodes']:
        if node['name'] == tgt_state_name:
            tgt_state = node
            break
    return tgt_state

def query_multi_edge():
    pass


def query_node(utg_file, tgt_state_name):
    tgt_state = None
    if isinstance(utg_file, str):
        with open(json_file_path, 'r') as utg_file:
            utg_data = json.load(utg_file)
    else:
        utg_data = utg_file

    for node in utg_data['nodes']:
        if node['name'] == tgt_state_name:
            tgt_state = node
            break
    return tgt_state



