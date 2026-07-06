# SPDX-License-Identifier: MIT

from jaxonomy.framework import DependencyTicket, next_dependency_ticket


def test_empty():
    ticket = DependencyTicket.nothing
    assert ticket == 0


def test_increments():
    ticket1 = next_dependency_ticket()
    assert ticket1 == DependencyTicket._next_available

    ticket2 = next_dependency_ticket()
    assert ticket2 == ticket1 + 1

    # Test inequality
    assert ticket1 != ticket2
    assert ticket1 < ticket2
    assert ticket2 > ticket1
