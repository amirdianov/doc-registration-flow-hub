from yargy import rule, or_
from yargy.predicates import dictionary, type as t1, eq
from yargy.pipelines import morph_pipeline


INT = t1('INT')

PUNKT = or_(
    eq(','),
    eq('.'),
    eq('/'),
    eq(':'),
    eq(';'),
    eq('-')
)

R_INN = rule(dictionary({'ИНН'}), PUNKT.optional(), INT)
R_SNILS = rule(dictionary({'СНИЛС'}), PUNKT.optional(), INT)


PRESudD = morph_pipeline([
    'дело',
    'гражданское дело',
    'производство'
])

NUMSudD = rule(
    or_(eq('M'), eq('m'), eq('м'), eq('М'), eq('2'), eq('02'), eq('9'), eq('0'), eq('13'), eq('11'), eq('СП7'),
        eq('сп7')),
    eq('-'),
    INT
)

ArbNUMSudD = rule(
    or_(eq('А'), eq('A'), eq('a'), eq('а')),
    INT,
    eq('-'),
    INT
)

PUNKT = or_(
    eq('/'),
    eq('-'),
    eq('('),
    eq(')')
)

R_SudD = or_(
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}), NUMSudD, PUNKT, INT),
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}), ArbNUMSudD, PUNKT, INT),
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}).optional(), NUMSudD, PUNKT,
         INT, PUNKT, INT),
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}).optional(), NUMSudD, PUNKT,
         INT),
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}), NUMSudD, PUNKT, INT, PUNKT,
         INT, dictionary({'М', 'м', 'M'}), PUNKT, INT, dictionary({'М', 'м', 'M'})),
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}), NUMSudD, PUNKT, INT, PUNKT,
         INT, dictionary({'М', 'м', 'M'})),
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}), NUMSudD, PUNKT, INT, PUNKT,
         INT),
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}), NUMSudD, PUNKT, INT, PUNKT,
         INT, PUNKT),
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}), NUMSudD, PUNKT, INT, PUNKT,
         INT, PUNKT, INT),
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}), NUMSudD, PUNKT, INT, PUNKT,
         PUNKT, INT),
    rule(PRESudD.optional(), dictionary({'номер', '№', 'Na', 'Nu', 'Nt', 'Ns', 'Nv', 'N'}), NUMSudD))



