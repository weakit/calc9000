from . import p


def test_seed():
    assert p('SeedRandom[1234]; RandomReal[]') == p('SeedRandom[1234]; RandomReal[]')
    assert p('SeedRandom["password"]; RandomReal[]') == p('SeedRandom["password"]; RandomReal[]')
    assert p('SeedRandom[4567]; {RandomInteger[10], RandomReal[]}') == \
           p('SeedRandom[4567]; {RandomInteger[10], RandomReal[]}')
    assert p('RandomReal[]') != p('RandomReal[]') != p('RandomReal[]')
    assert p('SeedRandom[4567]; {RandomInteger[10], RandomReal[]}')
    assert p('SeedRandom[1234]; RandomInteger[10]') == p('SeedRandom[1234]; RandomInteger[10]')
    assert p('SeedRandom[1234]; RandomInteger[10]') != p('RandomInteger[10]') != p('RandomInteger[10]')
    assert p('SeedRandom[1234]; RandomReal[]') == p('SeedRandom[1234]; RandomReal[]')
    assert p('SeedRandom[1234]; RandomReal[]') != p('RandomReal[]') != p('RandomReal[]')
    assert p('SeedRandom[]; RandomReal[]') != p('SeedRandom[]; RandomReal[]')


def test_block():
    assert p('BlockRandom[RandomReal[]]') == p('RandomReal[]')
    assert p('SeedRandom[123]; BlockRandom[RandomReal[]]') == p('RandomReal[]')
    assert p('SeedRandom[123]; BlockRandom[SeedRandom[10]; RandomReal[]]') != \
           p('RandomReal[]') == p('SeedRandom[123]; RandomReal[]')
