package com.example.test

import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.drawerlayout.widget.DrawerLayout
import androidx.fragment.app.Fragment
import com.google.android.material.navigation.NavigationView
import androidx.core.view.GravityCompat

class MainActivity : AppCompatActivity(), NavigationView.OnNavigationItemSelectedListener, Fragment1.OnFragmentInteractionListener {

    private lateinit var drawerLayout: DrawerLayout
    private lateinit var currentFragment: Fragment
    private lateinit var navigationView: NavigationView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val toolbar: Toolbar = findViewById(R.id.toolbar)
        setSupportActionBar(toolbar)

        drawerLayout = findViewById(R.id.drawer_layout)
        navigationView = findViewById(R.id.nav_view)
        navigationView.setNavigationItemSelectedListener(this)

        val toggle = ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close)
        drawerLayout.addDrawerListener(toggle)
        toggle.syncState()

        if (savedInstanceState == null) {
            replaceFragment(Fragment1())
            navigationView.setCheckedItem(R.id.nav_home)
        }
    }

    override fun onNavigationItemSelected(item: MenuItem): Boolean {
        var frag: Fragment? = null
        when (item.itemId) {
            R.id.nav_home -> frag = Fragment1()
            R.id.nav_gps_location -> frag = Fragment2()
            R.id.nav_sound_level -> frag = Fragment3()
        }

        if (frag != null) {
            replaceFragment(frag)
            drawerLayout.closeDrawers()
            return true
        } else {
            Log.d("MainActivity", "Unknown menu item clicked")
            return false
        }
    }

    override fun onFragmentInteraction(direction: Int) {
        // 0 for back, 1 for next
        val currentFragmentIndex = when (currentFragment) {
            is Fragment1 -> 0
            is Fragment2 -> 1
            is Fragment3 -> 2
            else -> -1
        }

        val nextFragmentIndex = if (direction == 1) {
            (currentFragmentIndex + 1) % 3 // Cycle through 0, 1, 2
        } else {
            (currentFragmentIndex - 1 + 3) % 3 // Cycle through 2, 1, 0 (handling negative results)
        }

        when (nextFragmentIndex) {
            0 -> replaceFragment(Fragment1())
            1 -> replaceFragment(Fragment2())
            2 -> replaceFragment(Fragment3())
        }
        // Uncheck previous item and check new one in drawer layout
        when (nextFragmentIndex) {
            0 -> navigationView.setCheckedItem(R.id.nav_home)
            1 -> navigationView.setCheckedItem(R.id.nav_gps_location)
            2 -> navigationView.setCheckedItem(R.id.nav_sound_level)
        }
    }

    private fun replaceFragment(fragment: Fragment) {
        currentFragment = fragment
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, fragment)
            .commit()
    }

    override fun onSupportNavigateUp(): Boolean {
        return drawerLayout.openDrawer(GravityCompat.START) || super.onSupportNavigateUp()
    }
}