from pm4py.objects.log.obj import EventLog


def convert_bool_to_int(log: EventLog, attribute_name: str) -> EventLog:
    for trace in log:
        if attribute_name in trace.attributes:
            value = trace.attributes[attribute_name]
            if isinstance(value, bool):
                if value:
                    trace.attributes[attribute_name] = 1
                else:
                    trace.attributes[attribute_name] = 0

            continue

        for event in trace:
            if attribute_name in event:
                value = event[attribute_name]
                if isinstance(value, bool):
                    if value:
                        event[attribute_name] = 1
                    else:
                        event[attribute_name] = 0
    return log
