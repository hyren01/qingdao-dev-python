# -*- coding:utf-8 -*-
"""
实体类，对应数据库中的t_event_task表。
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""


class EventTask(object):

    def __init__(self):
        self._task_id = ''
        self._model_name = ''
        self._tables_name = ''
        self._task_remark = ''
        self._epoch = None
        self._create_date = ''
        self._status = ''
        self._model_id = ''
        self._sample_start_date = ''
        self._sample_end_date = ''
        self._task_finish_date = ''
        self._task_finish_time = ''
        self._create_time = ''
        self._predict_end_date = ''

        self._event_type = ''
        self._model_dir = ''
        self._model_type = ''
        self._event = ''

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, task_id):
        self._task_id = task_id

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name

    @property
    def tables_name(self):
        return self._tables_name

    @tables_name.setter
    def tables_name(self, tables_name: str):
        self._tables_name = tables_name

    @property
    def task_remark(self):
        return self._task_remark

    @task_remark.setter
    def task_remark(self, task_remark):
        self._task_remark = task_remark

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def create_date(self):
        return self._create_date

    @create_date.setter
    def create_date(self, create_date):
        self._create_date = create_date

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, model_id):
        self._model_id = model_id

    @property
    def sample_start_date(self):
        return self._sample_start_date

    @sample_start_date.setter
    def sample_start_date(self, sample_start_date):
        self._sample_start_date = sample_start_date

    @property
    def sample_end_date(self):
        return self._sample_end_date

    @sample_end_date.setter
    def sample_end_date(self, sample_end_date):
        self._sample_end_date = sample_end_date

    @property
    def task_finish_date(self):
        return self._task_finish_date

    @task_finish_date.setter
    def task_finish_date(self, task_finish_date):
        self._task_finish_date = task_finish_date

    @property
    def task_finish_time(self):
        return self._task_finish_time

    @task_finish_time.setter
    def task_finish_time(self, task_finish_time):
        self._task_finish_time = task_finish_time

    @property
    def create_time(self):
        return self._create_time

    @create_time.setter
    def create_time(self, create_time):
        self._create_time = create_time

    @property
    def predict_end_date(self):
        return self._predict_end_date

    @predict_end_date.setter
    def predict_end_date(self, predict_end_date):
        self._predict_end_date = predict_end_date

    @property
    def event_type(self):
        return self._event_type

    @event_type.setter
    def event_type(self, event_type):
        self._event_type = event_type

    @property
    def model_dir(self):
        return self._model_dir

    @model_dir.setter
    def model_dir(self, model_dir):
        self._model_dir = model_dir

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, model_type):
        self._model_type = model_type

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, event):
        self._event = event
