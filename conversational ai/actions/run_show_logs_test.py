import sys
from pathlib import Path

# Ensure parent (conversational ai) is on sys.path so imports behave
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def run_test():
    import importlib
    actions_mod = importlib.import_module('actions.actions')

    class DummyDispatcher:
        def __init__(self):
            self.messages = []
        def utter_message(self, text=None, json_message=None, **kwargs):
            self.messages.append({'text': text})

    class DummyTracker:
        def __init__(self):
            self.sender_id = 'test'
            self.latest_message = {'text': ''}
            self.events = []

    dispatcher = DummyDispatcher()
    tracker = DummyTracker()

    act = actions_mod.ActionShowNetworkLogs()
    act.run(dispatcher, tracker, {})

    print('Dispatched messages:')
    for m in dispatcher.messages:
        print(m)


if __name__ == '__main__':
    run_test()
