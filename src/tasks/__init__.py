from src.tasks.ia import SingleIATask, NlIATask, CipherIATask, SDEIATask
from src.tasks.debate import SingleDebateTask, NlDebateTask, CipherDebateTask, SDEDebateTask
from src.tasks.workflow import SingleWorkflowTask, NlWorkflowTask, CipherWorkflowTask, SDEWorkflowTask 
from src.tasks.utils import update_hidden_dim

def method_to_task_cls(task_type, method):
    task_type_to_cls = {
        "debate": {
            "single": SingleDebateTask,
            "nl": NlDebateTask,
            "cipher": CipherDebateTask,
            "sde": SDEDebateTask,
        },
        "ia": {
            "single": SingleIATask,
            "nl": NlIATask,
            "cipher": CipherIATask,
            "sde": SDEIATask,
        },
        "workflow": {
            "single": SingleWorkflowTask,
            "nl": NlWorkflowTask,
            "cipher": CipherWorkflowTask,
            "sde": SDEWorkflowTask,
        }
    }
    if task_type not in task_type_to_cls:
        raise NotImplementedError(f"Task type {task_type} is not implemented.")
    if method not in task_type_to_cls[task_type]:
        raise NotImplementedError(f"Method {method} is not implemented for task type {task_type}.")
    return task_type_to_cls[task_type][method]