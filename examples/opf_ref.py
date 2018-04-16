import sys
import pfnet
import optalg

case = sys.argv[1]

parser = pfnet.Parser(case)

net = parser.parse(case)
net.show_components()

net.clear_flags()

net.set_flags("bus",
              ["variable", "bounded"], 
              "any",
              "voltage magnitude")

# Voltage angles
net.set_flags("bus",
              "variable",
              "not slack",
              "voltage angle")

# Generator powers
net.set_flags("generator",
              ["variable","bounded"],
              "not on outage",
              ["active power","reactive power"])

# Objective function
gen_cost = pfnet.Function("generation cost", 1., net)

# Constaints
acpf = pfnet.Constraint("AC power balance", net)
bounds = pfnet.Constraint("variable bounds", net)
th_limits = pfnet.Constraint("AC branch flow limits", net)

# Problem
problem = pfnet.Problem(net)
problem.add_function(gen_cost)
problem.add_constraint(acpf)
problem.add_constraint(bounds)
problem.add_constraint(th_limits)
problem.analyze()
problem.show()

# Solve
solver = optalg.opt_solver.OptSolverIpopt()
solver.solve(problem)

# Update network
