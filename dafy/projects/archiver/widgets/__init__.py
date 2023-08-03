import logging
import os

logger = logging.getLogger('widget')
logger.propagate = False
if not os.path.exists('./log_temp'):
    os.mkdir('./log_temp')
f_handler = logging.FileHandler('./log_temp/widget_log.log', mode = 'w')
f_handler.setFormatter(logging.Formatter('%(levelname)s : %(name)s : %(message)s : %(asctime)s : %(lineno)d'))
f_handler.setLevel(logging.DEBUG)
logger.addHandler(f_handler)
