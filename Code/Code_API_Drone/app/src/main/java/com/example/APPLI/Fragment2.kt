package com.example.APPLI

import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import org.osmdroid.views.MapView
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.overlay.Marker
import android.preference.PreferenceManager

private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

/**
 * Fragment qui affiche la position GPS sur une carte OpenStreetMap (osmdroid)
 * et les coordonnées détaillées dans un TextView.
 */
class Fragment2 : Fragment() {
    private val TAG = "Fragment2"
    private var param1: String? = null
    private var param2: String? = null

    private lateinit var gpsTextView: TextView
    private val sharedDataManager: SharedDataManager by activityViewModels()
    private var mapView: MapView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            param1 = it.getString(ARG_PARAM1)
            param2 = it.getString(ARG_PARAM2)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_2, container, false)

        // --- Initialisation d'osmdroid ---
        val context = requireContext().applicationContext
        org.osmdroid.config.Configuration.getInstance().load(context, PreferenceManager.getDefaultSharedPreferences(context))

        // --- Initialisation des composants graphiques (MapView et TextView) ---
        gpsTextView = view.findViewById(R.id.textView2)
        gpsTextView.text = "Localisation GPS\n\nEn attente de données..."

        mapView = view.findViewById(R.id.map)
        mapView?.setMultiTouchControls(true)
        mapView?.controller?.setZoom(15.0)

        // --- Observation des données GPS partagées ---
        sharedDataManager.gpsData.observe(viewLifecycleOwner) { gpsData ->
            Log.d(TAG, "Données GPS reçues pour traitement dans Fragment2: '$gpsData'")

            // Regex pour extraire les nombres (latitude, longitude, etc.) des données reçues
            val numberPattern = Regex("-?\\d+\\.?\\d*")
            val numbers = numberPattern.findAll(gpsData).map { it.value }.toList()

            // Affichage des coordonnées formatées en colonne
            val displayBuilder = StringBuilder()
            if (numbers.isNotEmpty()) {
                if (numbers.size >= 1) displayBuilder.append("Lat: ${numbers[0]}\n")
                if (numbers.size >= 2) displayBuilder.append("Lon: ${numbers[1]}\n")
                if (numbers.size >= 3) displayBuilder.append("Alt: ${numbers[2]}\n")
                if (numbers.size >= 4) displayBuilder.append("Autre: ${numbers[3]}\n")
            } else {
                displayBuilder.append("Aucune donnée GPS valide")
            }
            gpsTextView.text = displayBuilder.toString()

            // Mise à jour de la position sur la carte si les données sont valides
            if (numbers.size >= 2) {
                val lat = numbers[0].toDoubleOrNull()
                val lon = numbers[1].toDoubleOrNull()
                if (lat != null && lon != null) {
                    updateMapLocation(lat, lon)
                }
            }
        }

        Log.d(TAG, "Fragment2 onCreateView terminé")
        return view
    }

    /**
     * Met à jour la carte en centrant la vue sur la nouvelle position GPS
     * et en y plaçant un marqueur.
     */
    private fun updateMapLocation(lat: Double, lon: Double) {
        val geoPoint = GeoPoint(lat, lon)
        mapView?.controller?.setCenter(geoPoint)
        mapView?.overlays?.clear()
        val marker = Marker(mapView)
        marker.position = geoPoint
        marker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
        marker.title = "Position actuelle"
        mapView?.overlays?.add(marker)
        mapView?.invalidate()
    }

    companion object {
        /**
         * Factory method pour créer une nouvelle instance du fragment.
         */
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            Fragment2().apply {
                arguments = Bundle().apply {
                    putString(ARG_PARAM1, param1)
                    putString(ARG_PARAM2, param2)
                }
            }
    }
}