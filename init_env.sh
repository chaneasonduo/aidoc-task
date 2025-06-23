#!/bin/bash
script_path=$(dirname "$(readlink -f $0)")

echo "${script_path}"

cd "${script_path}" || exit

# determine whether .venv directory exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
    exit 0
fi

python -m venv .venv

#!/bin/bash
script_path=$(dirname "$(readlink -f "$0")")

echo "${script_path}"

cd "${script_path}" || exit

# 判断.venv目录是否存在
if [ -d ".venv" ]; then
    echo "虚拟环境已存在。"
    exit 0
fi

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
case "$(uname -s)" in
    Linux|Darwin)
        source .venv/bin/activate
        ;;
    MINGW64*|CYGWIN*)
        source .venv/Scripts/activate
        ;;
    *)
        echo "不支持的操作系统"
        exit 1
        ;;
esac

pip install -r requirements.txt

echo "虚拟环境已创建并激活。"
