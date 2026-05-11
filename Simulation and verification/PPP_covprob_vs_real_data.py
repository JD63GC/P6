from skyfield.api import load, Topos
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter
import statistics
import numpy as np
import random
random.seed(42)

# Aalborg koordinater
aalborg = Topos(latitude_degrees=57.0488,
                longitude_degrees=9.9217)



# Hent satellitdata (Starlink)
stations_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'

satellites = load.tle_file(stations_url)

print(f'Antal satellitter loaded: {len(satellites)}')



# Tidsskala
ts = load.timescale()

# Antal snapshots
NUM_SNAPSHOTS = 200

# tilfældige tidspunkter over 30 dage
START_DATE = datetime(2026, 1, 1)
END_DATE   = datetime(2026, 1, 31)


# Funktion til at finde synlige satellitter
# (og har rette frekvensressource) for et givet tidspunkt
def count_visible_satellites(t):

    visible = 0

    for sat in satellites:

        difference = sat - aalborg
        topocentric = difference.at(t)

        alt, az, distance = topocentric.altaz()

        # Satellitten skal være synlig
        if alt.degrees > 0:  # Minimum højde over horisonten 
            #(burde være 25 grader for at være realistisk)

            # Random resourceblok fra 1-20
            resource = random.randint(1, 20)

            # Kun resourceblok 1 beholdes
            if resource == 1:
                visible += 1

    return visible

# Lav 200 snapshots og tæl synlige satellitter

counts = []

time_span = END_DATE - START_DATE

for i in range(NUM_SNAPSHOTS):

    # tilfældig tid i intervallet
    random_seconds = random.uniform(0, time_span.total_seconds())

    random_time = START_DATE + timedelta(seconds=random_seconds)

    t = ts.utc(
        random_time.year,
        random_time.month,
        random_time.day,
        random_time.hour,
        random_time.minute,
        random_time.second
    )

    visible_count = count_visible_satellites(t)

    counts.append(visible_count)

    print(
        f"Snapshot {i+1}: "
        f"{random_time} -> "
        f"{visible_count} synlige satellitter"
    )

"""
print(statistics.mean(counts))
# Histogram / pindediagram

frequency = Counter(counts)

x = sorted(frequency.keys())
y = [frequency[val] for val in x]

plt.figure(figsize=(10,6))
plt.bar(x, y)

plt.xlabel('Antal synlige satellitter i snapshot')
plt.ylabel('Antal snapshots')
plt.title('Fordeling af synlige satellitter over Aalborg')

plt.grid(True)

plt.show()
"""

#---------------------------------------------------
#Poisson model
#---------------------------------------------------

from scipy.stats import poisson

# ---------------------------------------------------
# Histogram data
# ---------------------------------------------------

frequency = Counter(counts)

x = sorted(frequency.keys())
y = [frequency[val] for val in x]

# ---------------------------------------------------
# Estimer middelværdi
# ---------------------------------------------------

mu = np.mean(counts)

print(f"\nEstimeret middelværdi mu = {mu:.3f}")

# ---------------------------------------------------
# Poisson model
# ---------------------------------------------------

x_poisson = np.arange(min(x), max(x)+1)

poisson_probs = poisson.pmf(x_poisson, mu)

# Skaler til antal snapshots
poisson_expected = poisson_probs * NUM_SNAPSHOTS

# ---------------------------------------------------
# Plot
# ---------------------------------------------------

plt.figure(figsize=(10,6))

# Simulation histogram
plt.bar(x, y, alpha=0.7, label='Simulation')

# Poisson kurve
plt.plot(
    x_poisson,
    poisson_expected,
    linewidth=2,
    label=f'Poisson($\\mu$={mu:.2f})'
)

plt.xlabel('Antal synlige satellitter')
plt.ylabel('Antal snapshots')

plt.title(
    'Simulation vs. Poisson model\n'
    'Starlink satellitter over Aalborg'
)

plt.legend()

plt.grid(True)

plt.show()
