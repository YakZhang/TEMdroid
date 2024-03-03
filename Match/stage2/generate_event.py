

def generate_event(source_event,transferable_widget_pair):
    source_input = source_event['input']
    source_action = source_event['action']
    target_widget = transferable_widget_pair[1]
    target_event = {
        'widget':target_widget,
        'input':source_input,
        'action':source_action,
    }
    return target_event


def generate_test(source_events,transferable_widget_pairs):
    target_events = []
    for idx in range(len(source_events)):
        source_event = source_events[idx]
        transferable_widget_pair = transferable_widget_pairs[idx]
        target_event = generate_event(source_event,transferable_widget_pair)
        target_events.append(target_event)
    return target_events

def generate_oracle(source_oracle, transferable_widget_pair,target_xml):
    source_condition = source_oracle['condition']
    source_attribute = source_oracle['widget']
    target_widget = transferable_widget_pair[1]
    source_widget = transferable_widget_pair[0]
    if source_attribute == source_widget:
        target_oracle = {
            'attribute':target_widget,
            'condition':source_condition,
            'type':'widget_related'
        }
    else:
        attribute = get_attribute(target_xml, target_widget)
        target_oracle = {
            'attribute':attribute,
            'condition':source_condition,
            'type':'attribute_related'
        }
    return target_oracle

def get_attribute(target_xml,target_widget):
    attribute = target_xml[target_widget].text
    return attribute