import click
from dafy.core.util.path import *

@click.command()
@click.option('--app', default='ubmate',
              help="app to be launched: ubmate (default), xrv, ctr, superrod, viewer, archiver")
def dispatcher(app):
    if app == 'xrv':
        from dafy.projects.xrv.scripts.xrv_gui import main
    elif app == 'ctr':
        from dafy.projects.ctr.scripts.ctr_gui import main
    elif app == 'ubmate':
        from dafy.projects.ubmate.scripts.ubmate_gui import main
    elif app == 'superrod':
        from dafy.projects.superrod.scripts.superrod_gui import main
    elif app == 'viewer':
        from dafy.projects.viewer.scripts.viewer_gui import main
    elif app == 'archiver':
        from dafy.projects.archiver.scripts.archiver_gui import main
    else:
        def main():
            print(f'APP tag {app} is undefined! Should be one of these: ubmate (default), xrv, ctr, superrod, viewer, archiver')
    main()