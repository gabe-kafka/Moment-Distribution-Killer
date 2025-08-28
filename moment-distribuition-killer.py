import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Input collection (unchanged)
beam_length = float(input("Enter beam length (ft): ").strip())
supports_input = input("Enter support locations from left end (ft, comma-separated): ").strip()
support_positions = sorted([float(x.strip()) for x in supports_input.split(',') if x.strip()])
point_loads = []
point_locs = input("Enter point load locations (ft, comma-separated, or press Enter for none): ").strip()
if point_locs:
    for loc_str in point_locs.split(','):
        loc = float(loc_str.strip())
        if 0 <= loc <= beam_length:
            P_val = float(input(f"Enter point load at {loc} ft (kips): ").strip())
            point_loads.append((loc, P_val * 1000.0))
        else:
            print(f"Skipping load at {loc} ft (outside beam).", file=sys.stderr)
    point_loads.sort(key=lambda x: x[0])
uniform_loads = []
while True:
    dist_input = input("Enter uniform load (start, end, kips/ft) or press Enter to continue: ").strip()
    if not dist_input:
        break
    try:
        start, end, w = [float(x.strip()) for x in dist_input.split(',')]
        if 0 <= start < end <= beam_length and w != 0:
            uniform_loads.append((start, end, w * 1000.0))
    except ValueError:
        print("Invalid input format for uniform load.", file=sys.stderr)

# FEM Implementation
def fem_reactions(beam_length, supports, point_loads, uniform_loads, EI=1e8):
    """
    Compute reactions using FEM.
    Returns reactions in lbs at support locations.
    EI in lb-ft^2 (default 1e8 for numerical stability).
    """
    nodes = set([0.0, beam_length])
    nodes.update(supports)
    nodes.update([x for x, _ in point_loads])
    for a, b, _ in uniform_loads:
        nodes.add(a)
        nodes.add(b)
    nodes = sorted(nodes)
    min_nodes = 5
    min_length = 1e-3
    refined_nodes = []
    for i in range(len(nodes)-1):
        a, b = nodes[i], nodes[i+1]
        if b - a < min_length:
            continue
        num = max(2, int(min_nodes * (b - a) / beam_length))
        seg = np.linspace(a, b, num, endpoint=(i == len(nodes)-2)).tolist()
        refined_nodes.extend(seg)
    nodes = sorted(set(refined_nodes))
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    elements = [(nodes[i], nodes[i+1]) for i in range(n_nodes-1) if nodes[i+1] - nodes[i] >= min_length]
    
    # Log element lengths for debugging
    element_lengths = [b - a for a, b in elements]
    if element_lengths:
        min_L = min(element_lengths)
        if min_L < 1e-3:
            print(f"Warning: Smallest element length is {min_L:.6f} ft", file=sys.stderr)
    
    K_global = np.zeros((n_dof, n_dof))
    for a, b in elements:
        L = b - a
        if L < min_length:
            continue
        k_e = (EI / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        i = nodes.index(a)
        j = nodes.index(b)
        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        for m, dm in enumerate(dofs):
            for n, dn in enumerate(dofs):
                K_global[dm, dn] += k_e[m, n]
    if np.any(np.isinf(K_global)) or np.any(np.isnan(K_global)):
        raise ValueError("Stiffness matrix contains inf/NaN. Check node spacing or EI.")
    F_global = np.zeros(n_dof)
    for x, P in point_loads:
        if x in nodes:
            idx = nodes.index(x)
            F_global[2*idx] -= P
        else:
            idx = np.searchsorted(nodes, x)
            if idx > 0 and idx < len(nodes):
                x1, x2 = nodes[idx-1], nodes[idx]
                L = x2 - x1
                if L >= min_length:
                    F_global[2*(idx-1)] -= P * (x2 - x) / L
                    F_global[2*idx] -= P * (x - x1) / L
    for a, b, w in uniform_loads:
        for x1, x2 in elements:
            L = x2 - x1
            if x2 <= a or x1 >= b or L < min_length:
                continue
            a_eff = max(a, x1)
            b_eff = min(b, x2)
            if b_eff <= a_eff:
                continue
            le = b_eff - a_eff
            xi1 = (a_eff - x1) / L
            xi2 = (b_eff - x1) / L
            # Consistent load vector using integrals of shape functions
            # N1 = 1 - 3*xi^2 + 2*xi^3, integral dxi = xi - xi^3 + 0.5*xi^4
            int_N1 = lambda xi: xi - xi**3 + 0.5 * xi**4
            F_v1 = w * L * (int_N1(xi2) - int_N1(xi1))
            # N2/L = xi - 2*xi^2 + xi^3, integral dxi = 0.5*xi^2 - (2/3)*xi^3 + (1/4)*xi^4
            int_N2_L = lambda xi: 0.5 * xi**2 - (2/3) * xi**3 + (1/4) * xi**4
            F_theta1 = w * L**2 * (int_N2_L(xi2) - int_N2_L(xi1))
            # N3 = 3*xi^2 - 2*xi^3, integral dxi = xi^3 - 0.5*xi^4
            int_N3 = lambda xi: xi**3 - 0.5 * xi**4
            F_v2 = w * L * (int_N3(xi2) - int_N3(xi1))
            # N4/L = -xi^2 + xi^3, integral dxi = -(1/3)*xi^3 + (1/4)*xi^4
            int_N4_L = lambda xi: -(1/3) * xi**3 + (1/4) * xi**4
            F_theta2 = w * L**2 * (int_N4_L(xi2) - int_N4_L(xi1))
            i = nodes.index(x1)
            j = nodes.index(x2)
            F_global[2*i] -= F_v1
            F_global[2*i + 1] -= F_theta1
            F_global[2*j] -= F_v2
            F_global[2*j + 1] -= F_theta2
    support_dofs = []
    for s in supports:
        idx = min(range(len(nodes)), key=lambda i: abs(nodes[i] - s))
        if abs(nodes[idx] - s) < min_length:
            support_dofs.append(2*idx)
    support_dofs = sorted(set(support_dofs))
    free_dofs = [i for i in range(n_dof) if i not in support_dofs]
    K_red = K_global[np.ix_(free_dofs, free_dofs)]
    F_red = F_global[free_dofs]
    cond = np.linalg.cond(K_red)
    if cond > 1e12:
        print(f"Warning: Stiffness matrix condition number is {cond:.2e}", file=sys.stderr)
    try:
        u_red = np.linalg.solve(K_red, F_red)
    except np.linalg.LinAlgError:
        raise ValueError("Singular stiffness matrix. Check support placement or node spacing.")
    u_global = np.zeros(n_dof)
    for i, dof in enumerate(free_dofs):
        u_global[dof] = u_red[i]
    if np.any(np.isinf(u_global)) or np.any(np.isnan(u_global)):
        raise ValueError("Displacements contain inf/NaN. Check stiffness matrix or loads.")
    reactions = K_global @ u_global - F_global
    support_reactions = []
    for s in supports:
        idx = min(range(len(nodes)), key=lambda i: abs(nodes[i] - s))
        if abs(nodes[idx] - s) < min_length:
            support_reactions.append(reactions[2*idx])
        else:
            support_reactions.append(0.0)
    return support_reactions

# Shear Diagram Computation (unchanged)
def compute_shear_diagram(beam_length, supports, reactions, point_loads, uniform_loads):
    """
    Compute shear force V(x) in lbs at a dense set of x positions.
    Returns x (ft) and V (kips) arrays, and a table of key points.
    """
    key_points = set([0.0, beam_length])
    key_points.update(supports)
    key_points.update([x for x, _ in point_loads])
    for a, b, _ in uniform_loads:
        key_points.add(a)
        key_points.add(b)
    key_points = sorted(key_points)
    x = np.linspace(0, beam_length, 1000)
    for kp in key_points:
        if not any(np.isclose(x, kp, atol=1e-6)):
            idx = np.searchsorted(x, kp)
            x = np.insert(x, idx, kp)
    x = np.sort(x)
    V = np.zeros_like(x)
    for s, R in zip(supports, reactions):
        V += (x >= s) * R
    for xp, P in point_loads:
        V -= (x > xp) * P
    for a, b, w in uniform_loads:
        left = x > a
        dist = np.clip(np.minimum(x, b) - a, 0.0, b-a)
        V -= left * w * dist
    V_kips = V / 1000.0
    shear_table = []
    for kp in key_points:
        idx = np.argmin(np.abs(x - kp))
        shear_table.append((kp, V_kips[idx]))
    V_max = np.max(V_kips)
    V_min = np.min(V_kips)
    idx_max = np.argmax(V_kips)
    idx_min = np.argmin(V_kips)
    if not any(np.isclose(kp, x[idx_max], atol=1e-6) for kp, _ in shear_table):
        shear_table.append((x[idx_max], V_max))
    if not any(np.isclose(kp, x[idx_min], atol=1e-6) for kp, _ in shear_table):
        shear_table.append((x[idx_min], V_min))
    shear_table.sort(key=lambda t: t[0])
    return x, V_kips, shear_table

# Moment Diagram Computation
def compute_moment_diagram(x, V_kips):
    """
    Compute moment M(x) in kip-ft by integrating V(x).
    Returns M (kip-ft) array.
    """
    V_lbs = V_kips * 1000.0  # Convert to lbs for integration
    M = np.zeros_like(x)  # Moment in lb-ft
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        M[i] = M[i-1] + 0.5 * (V_lbs[i-1] + V_lbs[i]) * dx
    M_kipft = M / 1000.0  # Convert to kip-ft
    return M_kipft

# Plotting with Plotly
def plot_diagrams(beam_length, supports, point_loads, uniform_loads, x, V_kips, M_kipft):
    """
    Plot load, shear, and moment diagrams using Plotly.
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=["Load Diagram", "Shear Diagram (kips)", "Moment Diagram (kip-ft)"])
    
    # Load Diagram (Row 1)
    fig.add_trace(go.Scatter(x=[0, beam_length], y=[0, 0], mode='lines', line=dict(color='black', width=3), name='Beam'), row=1, col=1)
    fig.add_trace(go.Scatter(x=supports, y=[0]*len(supports), mode='markers', marker=dict(symbol='triangle-up', size=10, color='black'), name='Supports'), row=1, col=1)
    max_point_load = max((abs(P) for _, P in point_loads), default=1.0)
    scale = 0.5
    for loc, P in point_loads:
        direction = -1 if P > 0 else 1
        arrow_len = (abs(P) / max_point_load) * scale
        fig.add_trace(go.Scatter(x=[loc, loc], y=[0, direction * arrow_len], mode='lines', line=dict(color='red', width=2), name='Point Load'), row=1, col=1)
        fig.add_annotation(x=loc, y=direction * arrow_len, text=f'{P/1000:.2f} kips', showarrow=False, yshift=10 * direction, font=dict(size=10, color='red'), row=1, col=1)
    max_uniform_intensity = max((abs(w) for _, _, w in uniform_loads), default=1.0)
    for start, end, w in uniform_loads:
        direction = -1 if w > 0 else 1
        rel_intensity = abs(w) / max_uniform_intensity
        load_height = rel_intensity * scale
        xx = np.linspace(start, end, 50)
        yy = np.full_like(xx, direction * load_height)
        fig.add_trace(go.Scatter(x=xx, y=yy, fill='tozeroy', mode='none', fillcolor='rgba(0,0,255,0.3)', name='Uniform Load'), row=1, col=1)
        fig.add_annotation(x=(start+end)/2, y=direction * load_height, text=f'{w/1000:.2f} kips/ft', showarrow=False, yshift=10 * direction, font=dict(size=10, color='blue'), row=1, col=1)
    
    # Shear Diagram (Row 2)
    fig.add_trace(go.Scatter(x=x, y=V_kips, mode='lines', line=dict(color='green'), name='Shear V'), row=2, col=1)
    V_max = np.max(V_kips)
    V_min = np.min(V_kips)
    idx_max = np.argmax(V_kips)
    idx_min = np.argmin(V_kips)
    fig.add_annotation(x=x[idx_max], y=V_max, text=f'Max V: {V_max:.2f} kips', showarrow=False, yshift=10, font=dict(color='green', size=10), row=2, col=1)
    fig.add_annotation(x=x[idx_min], y=V_min, text=f'Min V: {V_min:.2f} kips', showarrow=False, yshift=-15, font=dict(color='green', size=10), row=2, col=1)
    
    # Moment Diagram (Row 3)
    fig.add_trace(go.Scatter(x=x, y=M_kipft, mode='lines', line=dict(color='magenta'), name='Moment M'), row=3, col=1)
    M_max = np.max(M_kipft)
    M_min = np.min(M_kipft)
    idx_max = np.argmax(M_kipft)
    idx_min = np.argmin(M_kipft)
    fig.add_annotation(x=x[idx_max], y=M_max, text=f'Max M: {M_max:.2f} kip-ft', showarrow=False, yshift=10, font=dict(color='magenta', size=10), row=3, col=1)
    fig.add_annotation(x=x[idx_min], y=M_min, text=f'Min M: {M_min:.2f} kip-ft', showarrow=False, yshift=-15, font=dict(color='magenta', size=10), row=3, col=1)
    
    # Layout
    fig.update_xaxes(title_text="Position along beam (ft)", row=3, col=1)
    fig.update_yaxes(title_text="Load", row=1, col=1, zeroline=True)
    fig.update_yaxes(title_text="Shear (kips)", row=2, col=1, zeroline=True)
    fig.update_yaxes(title_text="Moment (kip-ft)", row=3, col=1, zeroline=True)
    fig.update_layout(showlegend=False, height=800, width=800, margin=dict(l=50, r=50, t=50, b=50))
    fig.show()

# Compute reactions in lbs and convert to kips
reactions_lbs = fem_reactions(beam_length, support_positions, point_loads, uniform_loads)
reactions_kips = [r / 1000.0 for r in reactions_lbs]

# Print reactions
for s, r in zip(support_positions, reactions_kips):
    print(f"Reaction at x={s:.3f} ft: {r:.3f} kips")

# Compute shear and moment diagrams
x, V_kips, shear_table = compute_shear_diagram(beam_length, support_positions, reactions_lbs, point_loads, uniform_loads)
M_kipft = compute_moment_diagram(x, V_kips)

# Plot diagrams
plot_diagrams(beam_length, support_positions, point_loads, uniform_loads, x, V_kips, M_kipft)
