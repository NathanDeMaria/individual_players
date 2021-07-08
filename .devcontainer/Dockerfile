FROM python:3.9

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/root/.bashrc"

# Poetry
ENV POETRY_VERSION 1.1.5
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/${POETRY_VERSION}/get-poetry.py | python
ENV PATH /root/.poetry/bin:${PATH}

# Is this a good way to add stuff on load?
# Not sure, but it's working
COPY profile.sh /root/profile.sh
RUN echo ". profile.sh" >> "/root/.bashrc"

# Waiting on https://github.com/python-poetry/poetry/pull/1822
RUN pip install "git+https://github.com/NathanDeMaria/EndGame.git@bb844f31297dbb7b5e4e8eb9cb5fab17eae5830c#egg=endgame&subdirectory=py-endgame"

COPY poetry.* ./
COPY pyproject.toml ./
RUN poetry install
