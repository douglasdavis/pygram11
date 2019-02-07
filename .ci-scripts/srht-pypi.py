import subprocess
import pygram11.version
import sys

def main():
    res = subprocess.run(['git', 'describe'], stdout=subprocess.PIPE)
    describe_out = res.stdout.decode('utf-8').split('-')
    print(describe_out)
    if len(describe_out) > 1:
        return 0
    elif pygram11.version.version == describe_out[0].strip():
        res = subprocess.run('twine upload dist/*', shell=True)
        return res.returncode
    else:
        return 0;

if __name__ == '__main__':
    main()
    sys.exit()
