import json
import datetime
import sys
import pandas as pd
import numpy as np
import yfinance as yf

# Configuración de las acciones requeridas
ACCIONES_CONFIG = [
    {"ticker": "ECOPETROL.CL", "nombre": "Ecopetrol", "nemotecnico": "ECOPETROL", "sector": "Petróleo y gas"},
    {"ticker": "PFCIBEST.CL", "nombre": "Grupo Cibest preferencial", "nemotecnico": "PFCIBEST", "sector": "Financiero"},
    {"ticker": "CIBEST.CL", "nombre": "Grupo Cibest ordinaria", "nemotecnico": "CIBEST", "sector": "Financiero"},
    {"ticker": "ISA.CL", "nombre": "ISA", "nemotecnico": "ISA", "sector": "Energía e infraestructura"},
    {"ticker": "EXITO.CL", "nombre": "Grupo Éxito", "nemotecnico": "EXITO", "sector": "Comercio"},
    {"ticker": "PFAVAL.CL", "nombre": "Grupo Aval preferencial", "nemotecnico": "PFAVAL", "sector": "Financiero"},
    {"ticker": "GEB.CL", "nombre": "Grupo Energía Bogotá", "nemotecnico": "GEB", "sector": "Energía"},
    {"ticker": "CELSIA.CL", "nombre": "Celsia", "nemotecnico": "CELSIA", "sector": "Energía"}
]

def clean_num(val, round_decimals=4):
    """Limpia valores nulos/infinitos y redondea."""
    if pd.isna(val) or np.isinf(val):
        return 0.0
    return round(float(val), round_decimals)

def obtener_trm():
    """Descarga la TRM actual (USD a COP)."""
    try:
        df = yf.Ticker("USDCOP=X").history(period="1mo")
        if not df.empty:
            return {
                "valor": clean_num(df['Close'].iloc[-1], 2),
                "fecha": df.index[-1].strftime("%Y-%m-%d")
            }
    except Exception as e:
        print(f"Aviso: No se pudo obtener la TRM. Motivo: {e}")
    return None

def procesar_accion(acc):
    """Descarga datos y calcula métricas para una sola acción."""
    ticker = acc['ticker']
    # auto_adjust=False es clave para tener 'Close' original y 'Adj Close'
    df = yf.Ticker(ticker).history(period="3y", auto_adjust=False)
    
    if df.empty:
        raise ValueError("No se obtuvieron datos de Yahoo Finance.")
        
    # Validar columnas necesarias
    for col in ['Close', 'Adj Close']:
        if col not in df.columns:
            raise ValueError(f"Falta la columna {col} en los datos.")

    # 1. Precio y Fecha actual
    precio_actual = df['Close'].iloc[-1]
    fecha_actual = df.index[-1].strftime("%Y-%m-%d")
    
    # 2. Variación del día
    if len(df) >= 2:
        var_dia = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1
    else:
        var_dia = 0.0
        
    # 3. Ventana de 1 Año (aprox 365 días hacia atrás)
    fecha_hace_un_ano = df.index[-1] - pd.DateOffset(years=1)
    df_1a = df[df.index >= fecha_hace_un_ano]
    
    # 4. Rentabilidad 1 Año (Ajustada)
    if not df_1a.empty and len(df_1a) > 1:
        rent_1a = (df_1a['Adj Close'].iloc[-1] / df_1a['Adj Close'].iloc[0]) - 1
    else:
        rent_1a = 0.0
        
    # 5. Rentabilidad Anualizada 3 Años (Ajustada)
    dias_totales = (df.index[-1] - df.index[0]).days
    anios_transcurridos = dias_totales / 365.25
    if anios_transcurridos > 0:
        rent_3a = (df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0]) ** (1 / anios_transcurridos) - 1
    else:
        rent_3a = 0.0
        
    # 6. Volatilidad Anual y Caída Máxima (sobre datos del último año)
    if not df_1a.empty and len(df_1a) > 2:
        # Rendimientos logarítmicos diarios
        log_ret = np.log(df_1a['Adj Close'] / df_1a['Adj Close'].shift(1)).dropna()
        vol_anual = log_ret.std() * np.sqrt(252)
        
        # Caída máxima (Drawdown)
        picos_acumulados = df_1a['Adj Close'].cummax()
        drawdowns = (df_1a['Adj Close'] / picos_acumulados) - 1
        caida_max = drawdowns.min()
    else:
        vol_anual = 0.0
        caida_max = 0.0
        
    # 7. Rendimiento por dividendo (Dividend Yield)
    if 'Dividends' in df_1a.columns and not df_1a['Dividends'].empty:
        div_sum_1a = df_1a['Dividends'].sum()
        div_yield = div_sum_1a / precio_actual
    else:
        div_yield = 0.0
        
    # 8. Histórico (último precio de cierre de cada mes en los 3 años)
    df_hist = df.copy()
    df_hist['mes'] = df_hist.index.strftime("%Y-%m")
    mensual = df_hist.groupby('mes').last().reset_index()
    historico = [{"mes": row['mes'], "cierre": clean_num(row['Close'], 2)} for _, row in mensual.iterrows()]
    
    # Nota sobre formato: Se manejan como proporciones (ej. 0.1250) para asegurar compatibilidad 
    # con el formato Intl.NumberFormat(style: 'percent') del frontend.
    return {
        "ticker": ticker,
        "nombre": acc['nombre'],
        "nemotecnico": acc['nemotecnico'],
        "sector": acc['sector'],
        "moneda": "COP",
        "precio": clean_num(precio_actual, 2),
        "fecha": fecha_actual,
        "variacion_dia_pct": clean_num(var_dia, 4),
        "rent_1a_pct": clean_num(rent_1a, 4),
        "rent_anual_3a_pct": clean_num(rent_3a, 4),
        "volatilidad_anual_pct": clean_num(vol_anual, 4),
        "caida_maxima_1a_pct": clean_num(caida_max, 4),
        "dividendo_rend_pct": clean_num(div_yield, 4),
        "historico": historico
    }

def main():
    print("=== INICIANDO ACTUALIZACIÓN DE DATOS (BVC) ===")
    acciones_procesadas = []
    errores = []
    
    # 1. Obtener TRM
    trm_data = obtener_trm()
    
    # 2. Procesar cada acción de forma independiente
    for acc in ACCIONES_CONFIG:
        try:
            print(f"Descargando {acc['ticker']}...")
            datos = procesar_accion(acc)
            acciones_procesadas.append(datos)
        except Exception as e:
            print(f" -> Error en {acc['ticker']}: {str(e)}")
            errores.append({"ticker": acc['ticker'], "motivo": str(e)})
            
    # 3. Reporte final
    print("\n=== RESUMEN DE EJECUCIÓN ===")
    print(f"Acciones exitosas: {len(acciones_procesadas)}")
    print(f"Acciones fallidas: {len(errores)}")
    
    # 4. Guardar archivo si hay datos
    if len(acciones_procesadas) > 0:
        estructura_final = {
            "actualizado": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fuente": "Yahoo Finance (yfinance); rentabilidades con precio ajustado por dividendos",
            "trm": trm_data,
            "acciones": acciones_procesadas,
            "errores": errores
        }
        
        try:
            with open("datos.json", "w", encoding="utf-8") as f:
                json.dump(estructura_final, f, ensure_ascii=False, indent=2)
            print("Archivo 'datos.json' actualizado y guardado correctamente.")
        except Exception as e:
            print(f"Error crítico al intentar guardar el archivo JSON: {e}")
    else:
        print("AVISO: No se procesó ninguna acción con éxito. Se conserva el archivo datos.json anterior.")

    # Siempre termina sin lanzar excepciones que rompan el entorno (Exit 0)
    sys.exit(0)

if __name__ == "__main__":
    main()
