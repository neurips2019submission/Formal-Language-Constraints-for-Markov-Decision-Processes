package dk.brics.automaton;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by hd on 04/14/18.
 */
public class CustomAutomaton extends Automaton {
    private final String newline = System.getProperty("line.separator");
    private Automaton automaton;

    public CustomAutomaton(Automaton automaton) {
        super();
        this.automaton = automaton;
    }

    public void appendDot(StringBuilder b, String stateLetter, Transition transition) {
        String to = String.format(" -> \"%s%d\" [label=\"", stateLetter, transition.to.number);
        b.append(to);
        transition.appendCharString(transition.min, b);
        if (transition.min != transition.max) {
            b.append("-");
            transition.appendCharString(transition.max, b);
        }
        b.append("\"]" + newline);
    }

    public void assignStateNumbers(List<State> states) {
        if (states.size() == Integer.MAX_VALUE)
            throw new IllegalArgumentException("number of states exceeded Integer.MAX_VALUE");

        int number = 1;

        for (State s : states) {
            if (s == automaton.initial)
                s.number = 0;
            else
                s.number = number++;
        }
    }

    public void totalize() {
        for (State p : automaton.getStates()) {
            List<Transition> transitions = new ArrayList<>(p.getTransitions());
            p.resetTransitions();
            for (Transition t : transitions) {
                char min = t.getMin();
                char max = t.getMax();
                State dest = t.getDest();
                // adding all transitions explicitly
                while (min != (max + 1))
                    p.addTransition(new Transition(min++, dest));
            }
        }
    }

    public String toDot(String stateLetter) {
        StringBuilder b = new StringBuilder("digraph DFA {" + newline);
        b.append("  rankdir = LR;" + newline);


        /**
         * Sorting states according to their state number for a better layout (visualization purpose)
         */
        Set<State> states = automaton.getStates();
        List<State> stateList = states.stream().collect(Collectors.toList());
        stateList.sort(
                (State s1, State s2) -> Integer.compare(s1.id, s2.id)
        );
        assignStateNumbers(stateList);

        for (State s : stateList) {
            String state = String.format("  \"%s%d\"", stateLetter, s.number);
            b.append(state);
            if (s.accept) {
                String attr = String.format(" [shape=doublecircle, label=<%s<sub>%d</sub>>];%s", stateLetter, s.number, newline);
                b.append(attr);
            } else {
                String attr = String.format(" [shape=circle, label=<%s<sub>%d</sub>>];%s", stateLetter, s.number, newline);
                b.append(attr);
            }
//            if (s == automaton.initial) {
//                b.append("  initial [style = invis];" + newline);
//                String attr = String.format("  initial -> \"%s%d\"%s", stateLetter, s.number, newline);
//                b.append(attr);
//            }
            for (Transition t : s.getTransitions()) {
                String trans = String.format("  \"%s%d\"", stateLetter, s.number);
                b.append(trans);
                appendDot(b, stateLetter, t);
            }
        }

        return b.append("}" + newline).toString();
    }
}
