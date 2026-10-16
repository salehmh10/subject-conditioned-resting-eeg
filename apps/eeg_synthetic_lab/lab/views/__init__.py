"""Streamlit views operating exclusively on the prepared bundle."""
from . import overview,signal,fidelity,connectivity,identity,utility,methods,presentation

VIEWS={
    "Overview & models":overview.render,
    "Signal explorer":signal.render,
    "Fidelity explorer":fidelity.render,
    "Connectivity explorer":connectivity.render,
    "Identity & diversity":identity.render,
    "Classification utility":utility.render,
    "Learning & methods":methods.render,
}
