# 3.11.0-bullseye on 2022-10-30
FROM python@sha256:3e2e743ca3027c375a13ee7b9b0f7757b0091501a7556563afbc8000b5c862a5

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/root/.bashrc"

# Poetry
# git because I need https://github.com/python-poetry/poetry/pull/7367
RUN curl -sSL https://install.python-poetry.org | python3 - --git https://github.com/python-poetry/poetry.git@66d36520e984509d4adddf6446529b60924d089f
ENV PATH /root/.local/bin:${PATH}

# Is this a good way to add stuff on load?
# Not sure, but it's working
COPY profile.sh /root/profile.sh
RUN echo ". profile.sh" >> "/root/.bashrc"

COPY poetry.* ./
COPY pyproject.toml ./
RUN poetry install
