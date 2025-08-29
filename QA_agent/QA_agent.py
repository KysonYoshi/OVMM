openai_api_key = 'Your-OpenAI-API-Key-Here'
model_name = 'gpt-4o-mini' 

import openai
from openai import OpenAI
import copy
import numpy as np
import os
import sys

import ast
import astunparse
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
import heapq
import base64
from localization import build_msg_localizer

os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI()

class LMP:

    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg

        self._base_prompt = self._cfg['prompt_text']

        self._stop_tokens = list(self._cfg['stop'])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query, context=''):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session']:
            prompt += f'\n{self.exec_hist}'

        if context != '':
            prompt += f'\n{context}'

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{use_query}'

        return prompt, use_query

    def __call__(self, query, context='', **kwargs):
        prompt, use_query = self.build_prompt(query, context=context)
        messages = [{"role": "system", "content": "user are doing few-shot prompting. Please provide the Python code without enclosing it in triple backticks."},
                    {"role": "user", "content": prompt}]

        while True:
            try:
                code_str = client.chat.completions.create(
                    messages=messages,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['engine'],
                    max_tokens=self._cfg['max_tokens']
                )
                code_str = code_str.choices[0].message.content
                break
            except:
                print('Retrying after 10s.')
        if self._cfg['include_context'] and context != '':
            to_exec = f'{context}\n{code_str}'
            to_log = f'{context}\n{use_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{use_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f'LMP {self._name} exec:\n\n{to_log_pretty}\n')

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg['debug_mode']:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_exec}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            return lvars[self._cfg['return_val_name']]


class LMPFGen:

    def __init__(self, cfg, fixed_vars, variable_vars):
        self._cfg = cfg

        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg['prompt_text']

    def create_f_from_sig(self, f_name, f_sig, other_vars=None, fix_bugs=False, return_src=False):
        print(f'Creating function: {f_sig}')

        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}'
        prompt = f'{self._base_prompt}\n{use_query}'
        messages = [{"role": "system", "content": "user are doing few-shot prompting. Please provide the Python code without enclosing it in triple backticks."},
                    {"role": "user", "content": prompt}]

        while True:
            try:
                f_src = client.chat.completions.create(
                    messages=messages,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['engine'],
                    max_tokens=self._cfg['max_tokens']
                )
                f_src = f_src.choices[0].message.content
                break
            except:
                print('Retrying after 10s.')

        if fix_bugs:
            f_src = openai.Edit.create(
                model='gpt-4o-mini',
                input='# ' + f_src,
                temperature=0,
                instruction='Fix the bug if there is one. Improve readability. Keep same inputs and outputs. Only small changes. No comments.',
            )['choices'][0]['text'].strip()

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}
        
        exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        to_print = highlight(f'{use_query}\n{f_src}', PythonLexer(), TerminalFormatter())
        print(f'LMP FGEN created:\n\n{to_print}\n')

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(self, code_str, other_vars=None, fix_bugs=False, return_src=False):
        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(f_name, f_sig, new_fs, fix_bugs=fix_bugs, return_src=True)

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
                    lvars = {}
                    
                    exec_safe(f_src, gvars, lvars)
                    
                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):

    def __init__(self, fs, f_assigns):
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    #banned_phrases = ['import', '__']
    banned_phrases = []
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    exec(code_str, custom_gvars, lvars)

import json
from PIL import Image

class MultiviewSceneGraph():
    def __init__(self, video_id, msg_file_path):
        #init
        predicted_msg_file = msg_file_path + "/" + video_id + "/eval_results.json"
        self.localizer = build_msg_localizer(
                            msg_path = predicted_msg_file,
                            video_id = video_id,
                            experiment_mode="localize",
                            device = 0,
                            split = "real",
                        )
        self.video_id = video_id
        with open(predicted_msg_file, 'r', encoding='utf-8') as file:
            self.json_data = json.load(file)
            
    def map_uids_to_keys(self, obj_list):
        reverse_map = {}
        for key, uids in self.json_data['uidmap'].items():
            for uid in uids:
                reverse_map[uid] = key

        mapped_result = []
        for uid in obj_list:
            if uid in reverse_map:
                key = reverse_map[uid]
                mapped_result.append(key)

        return mapped_result
    
    def get_number_of_frames(self):
        return len(self.json_data["sampled_frames"])
    
    def get_frame2index(self, frame_number):
        return self.json_data['sampled_frames'].index(frame_number)
    
    def get_img2frame(self, img_name):
        img_path = './QA_agent/input_img/' + img_name + '.png'
        loc, _ = self.localizer.localize(img_path)
        return loc
    
    def get_index2frame(self, index):
        return self.json_data['sampled_frames'][index]

from PIL import Image
import matplotlib.pyplot as plt

class LMP_wrapper():
    def __init__(self, env, cfg, render=False):
        self.env = env
        self._cfg = cfg
        
    def get_frame2index(self, frame_number):
        return self.env.get_frame2index(frame_number)
    
    def get_img2frame(self, img_name):
        return self.env.get_img2frame(img_name)
    
    def get_env(self):
        return self.env
    
    def get_index2frame(self, index):
        return self.env.get_index2frame(index)
        
    def get_number_of_frames(self):
        return self.env.get_number_of_frames()
    
    def show_images_from_frames(self, frame_list):
        num_images = len(frame_list)
        num_cols = 3  
        num_rows = (num_images + num_cols - 1) // num_cols  

        plt.figure(figsize=(15, num_rows * 5))
        #image_paths = ["/mnt/NAS/data/jz4725/msgdata/Test" + '/' + self.env.video_id + '/' + self.env.video_id + '_frames/lowres_wide/' + self.env.video_id + '_' + frame_number + '.png' for frame_number in frame_list]
        image_paths = ["/home/cl6933/MSG/GDoutputs/real" + '/' + self.env.video_id + '/' + self.env.video_id + '_frames/lowres_wide/' + self.env.video_id + '_' + frame_number + '.png' for frame_number in frame_list]
        for i, image_path in enumerate(image_paths):
            if os.path.exists(image_path):
                image = Image.open(image_path)
                plt.subplot(num_rows, num_cols, i + 1)
                plt.imshow(image)
                plt.title(os.path.basename(image_path))
                plt.axis('off')  

        plt.tight_layout()
        plt.show()

    
    def count_objects(self, objects):
        object_count = {}
        for obj in objects:
            if obj in object_count:
                object_count[obj] += 1
            else:
                object_count[obj] = 1
        return object_count
    
    def get_object_from_frame(self, frame_name):
        object_item_list = {}
        for a_key, a_values in self.env.json_data['annotations'][frame_name].items():
            for u_key, u_values in self.env.json_data['uidmap'].items():
                if a_key in u_values:
                    object_item_list[a_key] = u_key
        return object_item_list
    
    def get_object(self, object_name):
        object_list = {}
        if object_name in self.env.json_data["uidmap"]:
            for item in self.env.json_data["uidmap"][object_name]:
                object_list[item] = object_name
        else:
            obj = self.issimilar(object_name, self.env.json_data["uidmap"])
            if obj != '':
                object_list = self.get_object(obj)
        return object_list
    
    def get_object_frames(self, object_list):
        place_list = []
        for key_to_find in object_list:
            time_stamps = [time_stamp for time_stamp, keys in self.env.json_data['annotations'].items() if key_to_find in keys]
            place_list = place_list + time_stamps

        return place_list
    
    def shortest_path(self, start, goal):
        p_p = np.array(self.env.json_data['p-p'])
        """predicted_msg_file = msg_file_path + "/" + video_id + "/eval_results.json"
        with open(predicted_msg_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)"""
        pp_sim = np.array(self.env.json_data['pp-sim'])
        #pp_sim = np.array(json_data['pp-sim'])

        # 逐元素相乘
        graph = p_p * pp_sim

        # 將小於 0 的數值設為 0
        graph[graph < 0] = 0

        n = len(graph)
        distances = {node: float('inf') for node in range(n)}
        distances[start] = 0
        priority_queue = [(0, start)]
        previous_nodes = {node: None for node in range(n)}

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node == goal:
                path = []
                while previous_nodes[current_node] is not None:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                path.append(start)
                return path[::-1]

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in enumerate(graph[current_node]):
                if weight > 0:
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous_nodes[neighbor] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor))

        return None  
    
    def get_object_uids_list(self, frame_name):
        object_item_list = {}
        for a_key, a_values in self.env.json_data['annotations'][frame_name].items():
            for u_key, u_values in self.env.json_data['uidmap'].items():
                if a_key in u_values:
                    object_item_list[a_key] = u_key
        return object_item_list
    
    def issimilar(self, obj, object_list):
        new_prompt = f'is {obj} in {object_list}'
        messages = [{"role": "system", "content": "user are asking if the given object things are in the object_list. Please just return the object name in object_list. if not, return "". For example: object = 'tv', object_list = {'bed': ['NB59gmIiC4u5h2Mw'], 'table': ['RnVg7UM3yU93OL1o', '53naDCpgHHmCVkxd'], 'cabinet': ['BOYx4gvUEXXzkHo0', 'FbEfcoVRieMmQ4IW'], 'tv_monitor': ['qJ0TKTnoAkhV0k0C']} This should return 'tv_monitor'. object = 'book', object_list = {'bed': ['NB59gmIiC4u5h2Mw'], 'table': ['RnVg7UM3yU93OL1o', '53naDCpgHHmCVkxd'], 'cabinet': ['BOYx4gvUEXXzkHo0', 'FbEfcoVRieMmQ4IW'], 'tv_monitor': ['qJ0TKTnoAkhV0k0C']} This should return ''  "},
                    {"role": "user", "content": new_prompt}]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content
    
    def vlm(self, frame_number, text):
        image_path = msg_file_path + "/" + video_id + "/" + video_id + '_frames/lowres_wide/' + video_id + '_' + frame_number + '.png'
        with open(image_path, "rb") as image_file:
            img = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
            {
              "role": "user",
              "content": [
                {"type": "text", "text": text},
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{img}",
                  },
                },
              ],
            }
          ],
          max_tokens=300,
        )
        return response.choices[0].message.content
with open("QA_agent/prompt/prompt_tabletop_ui.txt", "r") as file:
    prompt_tabletop_ui = file.read().strip()

with open("QA_agent/prompt/prompt_fgen.txt", "r") as file:
    prompt_fgen = file.read().strip()

cfg_tabletop = {
  'lmps': {
    'tabletop_ui': {
      'prompt_text': prompt_tabletop_ui,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# ',
      'query_suffix': '.',
      'stop': ['#', 'objects = ['],
      'maintain_session': True,
      'debug_mode': False,
      'include_context': True,
      'has_return': False,
      'return_val_name': 'ret_val',
    },
    'fgen': {
      'prompt_text': prompt_fgen,
      'engine': model_name,
      'max_tokens': 512,
      'temperature': 0,
      'query_prefix': '# define function: ',
      'query_suffix': '.',
      'stop': ['# define', '# example'],
      'maintain_session': False,
      'debug_mode': False,
      'include_context': True,
    }
  }
}


def setup_LMP(env, cfg_tabletop):
    # LMP env wrapper
    cfg_tabletop = copy.deepcopy(cfg_tabletop)
    LMP_env = LMP_wrapper(env, cfg_tabletop)    
    
    # creating APIs that the LMPs can interact with
    fixed_vars = {
        'np': np, 'heapq': heapq
    }
    variable_vars = {
      k: getattr(LMP_env, k)
      for k in [
         'get_env', 'get_index2frame', 'get_frame2index', 'get_img2frame', 'show_images_from_frames',
          'get_number_of_frames', 'count_objects', 'get_object_from_frame', 'get_object',
          'get_object_frames', 'shortest_path', 'vlm', 'issimilar'
      ]
    }
    variable_vars['say'] = lambda msg: print(f'robot says: {msg}')

    # creating the function-generating LMP
    lmp_fgen = LMPFGen(cfg_tabletop['lmps']['fgen'], fixed_vars, variable_vars)
    
    # creating other low-level LMPs
    variable_vars.update({
      k: LMP(k, cfg_tabletop['lmps'][k], lmp_fgen, fixed_vars, variable_vars)
      for k in []
    })
    # creating the LMP that deals w/ high-level language commands
    lmp_tabletop_ui = LMP(
      'tabletop_ui', cfg_tabletop['lmps']['tabletop_ui'], lmp_fgen, fixed_vars, variable_vars
    )

    return lmp_tabletop_ui

if __name__ == '__main__':
    from localization import build_msg_localizer
    video_id = "41069025"
    msg_file_path = "/home/cl6933/MSG/exp-results/aomsg/2024-05-14_22-26-52/Test"

    env = MultiviewSceneGraph(video_id, msg_file_path)
    lmp_tabletop_ui = setup_LMP(env, cfg_tabletop)

    # Question for the model to process
    #question = "How many frames are there in the ."
    #question = "what kind of objects are there in the frame number 3044.722 and also give me the quantities"
    #question = "show me where are the tables."
    #question = "Where can I put my laptop."
    #question = "I am tired where can I go to sleep"
    #question = "Find frame numbers nearby 3044.722."
    question = "My current position is in picture 'start', how can I go to picture 'goal'?"
    ##question = "what do we have in this space"
    #question = "are the tables in frame 3044.239 and frame 3105.730 the same?"
    #question = "are the tables in frame 3044.239 and frame 3044.722 the same?"
    #question = "does the table in frame number: 3044.239 appear in other frames?"
    #question = "does the tv_monitor in frame number: 3127.721 appear in other frames?"
    #question = "is there any tv in this space?"
    #question = "How many books are there."
    #question = "What's the color of the cabinet in picture 3123.722?"

    #add does object in frame number: appear in other frames

    user_input = question #@param {allow-input: true, type:"string"}

    lmp_tabletop_ui(user_input, f'')



    """1. "Add object comparison for specific video frames"
    2. "Add command to check whether object appearance in specific frames"
    3. "Add `issimilar` function for fuzzy object name matching"
    4. "Add VLM access."""
