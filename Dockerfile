FROM python:3.10.7-bullseye

ADD ./. /cadex/
WORKDIR /cadex/benchmark/
RUN python3 -m pip install -r requirements_exp.txt

CMD python3 run_exp.py && until python3 -c "from cfbench.cfbench import analyze_results; analyze_results('cadex')"; do sleep 10; done